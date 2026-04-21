import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
import json
import torch

from lm_eval import evaluator, utils
# API differs between lm-eval versions:
#   * paper env c9bbec6e: initialize_tasks() + include_path() module funcs; ALL_TASKS registry
#   * paper_env_fast 0.4.2: TaskManager(include_path=...) passed to simple_evaluate
try:
    from lm_eval.tasks import initialize_tasks, include_path
    from lm_eval.api.registry import ALL_TASKS
    _USE_OLD_TASKS_API = True
except ImportError:
    from lm_eval.tasks import TaskManager
    initialize_tasks = None
    include_path = None
    ALL_TASKS = None
    _USE_OLD_TASKS_API = False

from utils_paper.process_args import process_args
from utils_paper.data import set_seed

from accelerate import Accelerator
accelerator = Accelerator()


def _install_llama_method(method, calib_path=None):
    """Swap models_paper.llama_kivi.LlamaForCausalLM_KIVI for the method
    variant so the existing LMEval wrapper picks it up via from_pretrained.
    Returns the restore function (used to undo the swap)."""
    import models_paper.llama_kivi as _m
    original = _m.LlamaForCausalLM_KIVI
    if method == "fp8":
        from models_paper.llama_kivi_fp8 import LlamaForCausalLM_FP8 as cls
    elif method == "pertoken":
        from models_paper.llama_kivi_pertoken import LlamaForCausalLM_KIVI_PerToken as cls
    elif method == "smoothkv":
        from models_paper.llama_smoothkv import LlamaForCausalLM_SmoothKV as cls
        assert calib_path is not None, "--calib_path required for smoothkv"
        cls._calib = torch.load(calib_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"unknown method {method!r}")
    _m.LlamaForCausalLM_KIVI = cls
    return lambda: setattr(_m, "LlamaForCausalLM_KIVI", original)


def _install_mistral_method(method, calib_path=None):
    import models_paper.mistral_kivi as _m
    original = _m.MistralForCausalLM_KIVI
    if method == "fp8":
        from models_paper.mistral_kivi_fp8 import MistralForCausalLM_FP8 as cls
    elif method == "pertoken":
        from models_paper.mistral_kivi_pertoken import MistralForCausalLM_KIVI_PerToken as cls
    elif method == "smoothkv":
        from models_paper.mistral_smoothkv import MistralForCausalLM_SmoothKV as cls
        assert calib_path is not None, "--calib_path required for smoothkv"
        cls._calib = torch.load(calib_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"unknown method {method!r}")
    _m.MistralForCausalLM_KIVI = cls
    return lambda: setattr(_m, "MistralForCausalLM_KIVI", original)


def _method_tag(args):
    """Build the output filename tag based on method + bits + group + residual."""
    if args.k_bits == 16:
        return "_fp16"
    m = args.method
    if m == "kivi":
        return f"_kivi{args.k_bits}bit_g{args.group_size}_res{args.residual_length}"
    if m == "fp8":
        return f"_fp8paper_g{args.group_size}"
    if m == "pertoken":
        return f"_pertokenpaper_int{args.k_bits}_g{args.group_size}_res{args.residual_length}"
    if m == "smoothkv":
        # Include calibration-parameter tags so α sweep / β sweep / percentile
        # sweep results don't collide in output filenames.
        tag_extra = ""
        if args.calib_path:
            import re
            a    = re.search(r"_a(\d+(?:\.\d+)?)",    args.calib_path)
            b    = re.search(r"_b(\d+(?:\.\d+)?)",    args.calib_path)
            pair = re.search(r"_pairK(\d+p?\d*)",     args.calib_path)
            pK   = re.search(r"(?<!pair)_pK(\d+p?\d*)", args.calib_path)
            pV   = re.search(r"_pV(\d+p?\d*)",        args.calib_path)
            # α/β variants can carry a "_pair" suffix indicating pair-max s_K
            alpha_pair = bool(re.search(r"_a\d+(?:\.\d+)?(?:_b\d+(?:\.\d+)?)?_pair(?!K)",
                                        args.calib_path))
            if a:    tag_extra += f"_a{a.group(1)}"
            if b:    tag_extra += f"_b{b.group(1)}"
            if alpha_pair: tag_extra += "_pair"
            if pair: tag_extra += f"_pairK{pair.group(1)}"
            elif pK: tag_extra += f"_pK{pK.group(1)}"
            if pV:   tag_extra += f"_pV{pV.group(1)}"
        return f"_smoothkvpaper_g{args.group_size}{tag_extra}"
    return f"_{m}"


if __name__ == "__main__":
    set_seed(42)

    model_args, data_args, training_args = process_args()
    dtype = torch.float16
    model_path = model_args.model_name_or_path.lower()
    is_llama = "llama" in model_path
    is_mistral = "mistral" in model_path

    low_cpu_mem_usage = True
    method = model_args.method if model_args.k_bits != 16 else "kivi"

    # For modern methods, monkey-patch the module so the existing LMEval wrapper
    # loads the right model class (and SmoothKV calibration).
    restore = None
    if method in ("fp8", "pertoken", "smoothkv"):
        if is_llama:
            restore = _install_llama_method(method, model_args.calib_path)
        elif is_mistral:
            restore = _install_mistral_method(method, model_args.calib_path)

    try:
        if is_llama:
            if model_args.k_bits == 16 and model_args.v_bits == 16:
                from models_paper.modeling_llama import LMEvalLlamaForCausalLM
                model = LMEvalLlamaForCausalLM(
                    k_bits=model_args.k_bits,
                    v_bits=model_args.v_bits,
                    group_size=model_args.group_size,
                    residual_length=model_args.residual_length,
                    pretrained=model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    dtype=dtype,
                    batch_size=data_args.batch_size,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
            else:
                from models_paper.llama_kivi import LMEvalLlamaForCausalLM_KIVI
                model = LMEvalLlamaForCausalLM_KIVI(
                    k_bits=model_args.k_bits if method == "kivi" else 4,  # placeholder for non-kivi
                    v_bits=model_args.v_bits if method == "kivi" else 4,
                    group_size=model_args.group_size,
                    residual_length=model_args.residual_length,
                    pretrained=model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    dtype=dtype,
                    batch_size=data_args.batch_size,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
        elif is_mistral:
            if model_args.k_bits == 16 and model_args.v_bits == 16:
                from models_paper.modeling_mistral import LMEvalMistralForCausalLM
                # vendored-code inconsistency: this wrapper takes buffer_length, not residual_length
                model = LMEvalMistralForCausalLM(
                    k_bits=model_args.k_bits,
                    v_bits=model_args.v_bits,
                    group_size=model_args.group_size,
                    buffer_length=model_args.residual_length,
                    pretrained=model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    dtype=dtype,
                    batch_size=data_args.batch_size,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    use_fast_tokenizer=False,
                )
            else:
                from models_paper.mistral_kivi import LMEvalMistralForCausalLM_KIVI
                model = LMEvalMistralForCausalLM_KIVI(
                    k_bits=model_args.k_bits if method == "kivi" else 4,
                    v_bits=model_args.v_bits if method == "kivi" else 4,
                    group_size=model_args.group_size,
                    residual_length=model_args.residual_length,
                    pretrained=model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    dtype=dtype,
                    batch_size=data_args.batch_size,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    use_fast_tokenizer=False,
                )
        else:
            raise NotImplementedError(
                f"Model {model_args.model_name_or_path} not supported. Use Llama or Mistral."
            )
    finally:
        if restore is not None:
            restore()

    if data_args.compile:
        # Compile the underlying HF model (model._model is the actual nn.Module,
        # the outer `model` is an HFLM/LMEvalLlamaForCausalLM wrapper).
        print(f"[compile] torch.compile(mode='reduce-overhead', dynamic=True, fullgraph=False)")
        model._model = torch.compile(model._model, mode="reduce-overhead",
                                      dynamic=True, fullgraph=False)

    if data_args.tasks is not None:
        # Register ALL local task dirs (math500, gsm8k_32k, gpqa_*_32k, aime*), not just math500.
        tasks_dir = os.path.join(os.path.dirname(__file__), "tasks")
        tm = None
        if _USE_OLD_TASKS_API:
            initialize_tasks()
            try:
                include_path(tasks_dir)
            except Exception as e:
                print(f"[warn] include_path {tasks_dir} failed: {e}")
            all_names = ALL_TASKS
        else:
            tm = TaskManager(include_path=tasks_dir)
            all_names = tm.all_tasks
        tasks_list = data_args.tasks.split(",")
        task_names = utils.pattern_match(tasks_list, all_names)
        for task in [task for task in tasks_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in tasks_list if task not in task_names and "*" not in task
        ]
        if task_missing:
            raise ValueError(
                f"Tasks {', '.join(task_missing)} were not found. "
                "Try `lm-eval --tasks list` for list of available tasks."
            )
        eval_kwargs = dict(model=model, tasks=task_names, log_samples=False)
        if tm is not None:
            eval_kwargs["task_manager"] = tm
        if data_args.max_gen_toks is not None:
            # simple_evaluate accepts gen_kwargs as "k=v,k=v" string
            eval_kwargs["gen_kwargs"] = f"max_gen_toks={data_args.max_gen_toks}"
        results = evaluator.simple_evaluate(**eval_kwargs)
        # make_table lives in evaluator (c9bbec6e) or utils (0.4.2)
        _make_table = getattr(evaluator, "make_table", None) or utils.make_table
        print(_make_table(results))

        os.makedirs("logs", exist_ok=True)
        model_short = model_args.model_name_or_path.rstrip("/").split("/")[-1].lower()
        out_name = f"{'_'.join(tasks_list)}_{model_short}{_method_tag(model_args)}_paper_results.json"
        out_path = os.path.join("logs", out_name)
        with open(out_path, "w") as f:
            json.dump(results["results"], f, indent=2)
        print(f"\nSaved: {out_path}")
