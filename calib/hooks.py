"""Hooks for capturing post-RoPE Q/K and pre-attention V from a HuggingFace
Transformers model during calibration.

K is captured **after RoPE** (because the actual K that hits the KV cache /
attention math is post-RoPE; quantizing pre-RoPE K would compute the wrong
``s_K`` for SmoothKV). V is captured pre-attention via a ``v_proj`` forward
hook — V has no RoPE.

The post-RoPE K capture works by monkey-patching each architecture's
``apply_rotary_pos_emb`` function: it returns Q, K rotated, and we intercept
the return values and forward them to the StatCollector. To know which layer
the rotated tensors came from, we wrap each ``self_attn.forward`` to set a
shared ``current_layer`` index before the RoPE call.
"""
import torch


def _get_layers(model):
    """Return the transformer layer ModuleList, walking past common wrappers.
    Handles plain ForCausalLM (model.model.layers), AutoModel base
    (model.layers), and multimodal wrappers like Exaone4_5_Model that hold
    the LM under .language_model (model.language_model.layers)."""
    for attr_chain in (("model", "layers"), ("language_model", "layers"),
                       ("model", "language_model", "layers"), ("layers",)):
        m = model
        ok = True
        for a in attr_chain:
            if not hasattr(m, a):
                ok = False
                break
            m = getattr(m, a)
        if ok:
            return m
    raise AttributeError(
        f"Could not find transformer layers on {type(model).__name__}; "
        f"tried .model.layers, .language_model.layers, .model.language_model.layers, .layers"
    )


def install_v_hook(model, collector):
    """Hook each layer's ``v_proj`` to capture V (pre-attention).

    V has no RoPE, so the v_proj output is exactly what the attention kernel
    receives. Returns a list of hook handles (call ``.remove()`` after
    calibration).
    """
    hooks = []
    for i, layer in enumerate(_get_layers(model)):
        def make_v_hook(layer_idx):
            def hook_v(module, inp, out):
                # out: (B, T, num_kv_heads * head_dim)
                B, T, _ = out.shape
                nh = collector.num_kv_heads
                D = collector.head_dim
                v = out.view(B, T, nh, D).transpose(1, 2).contiguous()
                collector.update_v(layer_idx, v)
            return hook_v
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))
    return hooks


def install_nope_qk_hook(model, collector):
    """Capture Q/K via q_norm / k_norm forward hooks for layers that skip
    ``apply_rotary_pos_emb`` (hybrid-attention models with global-NoPE
    layers, e.g. EXAONE-4.5: sliding-window layers apply RoPE, global
    layers don't).

    For NoPE layers, the K that hits the cache is the q_norm/k_norm output
    (no rotation applied) — so hooking post-norm captures the actual cached
    K. The existing ``monkey_patch_rope`` covers RoPE layers; this fills
    the gap for the rest.

    Only installs on a layer when ALL of:
      - ``self_attn.is_sliding`` exists (hybrid-attention attr)
      - ``is_sliding == False``
      - ``self_attn.sliding_window`` is not None (model uses sliding_window;
        global = NoPE branch)
      - ``self_attn.q_norm`` and ``self_attn.k_norm`` exist

    Non-hybrid architectures (Qwen3 / Llama / Mistral) fail the attribute
    check and the hook is silently skipped — their existing post-RoPE
    capture is unchanged.

    Returns hook handles.
    """
    hooks = []
    for i, layer in enumerate(_get_layers(model)):
        attn = layer.self_attn
        is_hybrid_nope = (
            hasattr(attn, 'is_sliding')
            and not attn.is_sliding
            and getattr(attn, 'sliding_window', None) is not None
            and hasattr(attn, 'q_norm')
            and hasattr(attn, 'k_norm')
        )
        if not is_hybrid_nope:
            continue

        def make_qk_hook(layer_idx, which):
            def hook(module, inp, out):
                # In EXAONE-4.5, q_norm/k_norm are applied to tensors of
                # shape (B, num_heads, T, D) — already reshaped/transposed
                # before the norm in Attention.forward.
                if which == 'q':
                    collector.update_q(layer_idx, out)
                else:
                    collector.update_k(layer_idx, out)
            return hook

        hooks.append(attn.q_norm.register_forward_hook(make_qk_hook(i, 'q')))
        hooks.append(attn.k_norm.register_forward_hook(make_qk_hook(i, 'k')))
    return hooks


def monkey_patch_rope(model, collector):
    """Capture post-RoPE Q and K by wrapping each architecture's
    ``apply_rotary_pos_emb`` and tracking the active layer index.

    Patches every Transformers attention module we know about (mistral / llama /
    qwen2 / qwen3 / qwen3_moe / exaone4) — only those that successfully imported
    are patched, so unknown architectures simply skip K capture (V still works
    via ``install_v_hook``).
    """
    # Track the active layer index across the apply_rotary_pos_emb wrap.
    current_layer = [0]

    def wrap_forward(layer_idx, orig_fwd):
        def new_fwd(*args, **kwargs):
            current_layer[0] = layer_idx
            return orig_fwd(*args, **kwargs)
        return new_fwd

    for i, layer in enumerate(_get_layers(model)):
        layer.self_attn.forward = wrap_forward(i, layer.self_attn.forward)

    def _make_patch(orig):
        # transformers 4.x: apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
        # transformers 5.x: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)   # position_ids removed
        # Forward *args/**kwargs verbatim so we don't introduce spurious positional args.
        def patched(*args, **kwargs):
            qr, kr = orig(*args, **kwargs)
            li = current_layer[0]
            collector.update_q(li, qr)
            collector.update_k(li, kr)
            return qr, kr
        return patched

    # Lazy import each architecture; skip ones not installed in this transformers build.
    targets = []
    for arch_name, modeling_path in [
        ("mistral",   "transformers.models.mistral.modeling_mistral"),
        ("llama",     "transformers.models.llama.modeling_llama"),
        ("qwen3",     "transformers.models.qwen3.modeling_qwen3"),
        ("qwen3_moe", "transformers.models.qwen3_moe.modeling_qwen3_moe"),
        ("qwen2",     "transformers.models.qwen2.modeling_qwen2"),
        ("exaone4",   "transformers.models.exaone4.modeling_exaone4"),
        ("exaone4_5", "transformers.models.exaone4_5.modeling_exaone4_5"),
        ("exaone_moe","transformers.models.exaone_moe.modeling_exaone_moe"),
    ]:
        try:
            mod = __import__(modeling_path, fromlist=["apply_rotary_pos_emb"])
            targets.append((arch_name, mod))
        except Exception:
            pass

    for _name, _mod in targets:
        _mod.apply_rotary_pos_emb = _make_patch(_mod.apply_rotary_pos_emb)
