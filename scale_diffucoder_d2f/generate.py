import argparse
import json
import os
from os import PathLike
import sys
import time

eval_plus_path = os.path.dirname(os.path.abspath(__file__)) + "/evalplus/"
sys.path = [eval_plus_path] + sys.path
from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


MODEL_MAPPING = {
    #  Can be either repo's name or /path/to/model
    "codeqwen": {
        "base": "Qwen/CodeQwen1.5-7B",
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
        "chat-awq": "Qwen/CodeQwen1.5-7B-Chat-AWQ",
    },
    "qwen2": {
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
    },
    "diffucoder": {
        "chat": {
            "base_model": os.environ.get("DIFFUCODER_BASE_MODEL", ""),
            "lora_path": os.environ.get("DIFFUCODER_LORA_PATH", ""),
        },
    },
    "diffucoder_basic": {
        "chat": {
            "base_model": os.environ.get("DIFFUCODER_BASE_MODEL", ""),
        },
    },
    "diffucoder_parallel": {
        "chat": {
            "base_model": os.environ.get("DIFFUCODER_BASE_MODEL", ""),
            "lora_path": os.environ.get("DIFFUCODER_LORA_PATH", ""),
        },
    },
}


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def code_generate(args, workdir: PathLike, model: DecoderBase, id_range=None):
    model.reset_statistics()
    run_start = time.time()
    responses = []

    with Progress(
        TextColumn(f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            dataset = get_human_eval_plus()
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            dataset = get_mbpp_plus()

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            if args.contract_type != "none" and task["contract"] == "":
                continue
            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len([f for f in os.listdir(os.path.join(workdir, p_name)) if f.endswith(".py")])
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            while sidx < args.n_samples:
                model.dataset = args.dataset
                outputs = model.codegen(
                    construct_contract_prompt(task["prompt"], args.contract_type, task["contract"]).strip(),
                    do_sample=not args.greedy,
                    num_samples=args.n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    if "```" in impl:
                        impl = impl.split("```")[0]
                        print("``` exist in generation. Please check the generation results.")
                    responses.append(impl)

                    try:
                        with open(
                            os.path.join(workdir, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            if model.direct_completion:
                                f.write(task["prompt"] + impl)
                            else:
                                f.write(impl)
                    except UnicodeEncodeError:
                        continue
                    sidx += 1

    stats = model.get_statistics()
    total_samples = stats.get("total_samples", 0)
    total_tokens = stats.get("total_generated_tokens", 0)
    total_actual_tokens = stats.get("total_actual_tokens_excluding_eos", 0)
    total_forward = stats.get("total_forward_passes", 0)
    tracked_time = stats.get("total_generation_time", 0.0)
    wall_time = time.time() - run_start
    effective_time = tracked_time if tracked_time > 0 else wall_time

    avg_tokens = total_tokens / total_samples if total_samples else 0.0
    avg_throughput = total_tokens / effective_time if effective_time > 0 else 0.0
    # forward_ratio = total_forward / total_tokens if total_tokens > 0 else 0.0
    forward_ratio =total_tokens / total_forward if total_forward > 0 else 0.0
    metrics = {
        "dataset": args.dataset,
        "model": str(model),
        "total_samples": total_samples,
        "total_generated_tokens": total_tokens,
        "total_actual_tokens_excluding_eos": total_actual_tokens,
        "total_forward_passes": total_forward,
        "total_generation_time": tracked_time,
        "wall_time_seconds": wall_time,
        "average_tokens_per_sample": avg_tokens,
        "average_throughput_tokens_per_second": avg_throughput,
        "forward_tokens_per_step": forward_ratio,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metrics_path = os.path.join(workdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    print(f"Saved generation metrics to {metrics_path}")

    save_dir = args.save_dir or workdir
    os.makedirs(save_dir, exist_ok=True)

    responses_path = os.path.join(save_dir, "rank_0_responses.jsonl")
    with open(responses_path, "w", encoding="utf-8") as responses_file:
        for response in responses:
            responses_file.write(json.dumps(response, ensure_ascii=False) + "\n")

    rank_stats = {
        "processed_samples": int(total_samples),
        "total_samples": int(total_samples),
        "total_generated_tokens_including_eos": int(total_tokens),
        "total_actual_tokens_excluding_eos": int(total_actual_tokens),
        "total_parallel_steps": int(total_forward),
        "total_time": wall_time,
        "generated_tokens_including_eos_per_second": float(total_tokens) / wall_time if wall_time > 0 else 0.0,
        "actual_tokens_excluding_eos_per_second": float(total_actual_tokens) / wall_time if wall_time > 0 else 0.0,
        "generated_tokens_including_eos_per_step": float(total_tokens) / float(total_forward) if total_forward > 0 else 0.0,
        "actual_tokens_excluding_eos_per_step": float(total_actual_tokens) / float(total_forward) if total_forward > 0 else 0.0,
        "timestamp": time.time(),
        "rank": 0,
        "world_size": 1,
    }

    stats_path = os.path.join(save_dir, "rank_0_stats.json")
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(rank_stats, stats_file, ensure_ascii=False, indent=2)

    print(f"Saved rank-0 responses to {responses_path}")
    print(f"Saved rank-0 stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, choices=MODEL_MAPPING.keys())
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_size", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-tag", type=str, default="", help="Optional suffix appended to output directory name.")
    # Parallel decoder hyperparameters (used when selecting the parallel decoder)
    parser.add_argument("--parallel-branching-factor", type=int, default=2)
    parser.add_argument("--parallel-branch-topp", type=float, default=0.5)
    parser.add_argument("--parallel-selection-alpha", type=float, default=0.5)
    parser.add_argument("--parallel-top-p", type=float, default=None)
    parser.add_argument("--parallel-top-k", type=int, default=None)
    parser.add_argument("--parallel-disable-verification", action="store_true")
    parser.add_argument("--parallel-disable-base-competition", action="store_true")
    parser.add_argument("--parallel-force-base-winner", action="store_true")
    parser.add_argument("--parallel-disable-uncertainty", action="store_true")

    # [NEW] Add arguments for DiffuCoder parameters
    parser.add_argument("--block-size", type=int, default=32, help="Block size for diffusion decoding")
    parser.add_argument("--block-add-threshold", type=float, default=0.3, help="Threshold to add new block")
    parser.add_argument("--skip-threshold", type=float, default=0.95, help="Confidence threshold to skip decoding")
    parser.add_argument("--decoded-token-threshold", type=float, default=0.95, help="Threshold to mark block as complete")
    parser.add_argument("--basic-token-per-step", type=int, default=1, help="Tokens generated per step for diffucoder_basic")

    args = parser.parse_args()
    print(args)
    assert args.model_size in MODEL_MAPPING[args.model_type]

    model_cfg = MODEL_MAPPING[args.model_type][args.model_size]
    model_path = args.model_path

    if model_path is None:
        if isinstance(model_cfg, dict):
            # diffucoder_basic expects only a base model path (no LoRA)
            if args.model_type == "diffucoder_basic":
                base_model = model_cfg.get("base_model", "")
                if not base_model:
                    raise ValueError(
                        "DiffuCoder basic requires base_model. Provide via --model_path or DIFFUCODER_BASE_MODEL env var."
                    )
                model_path = base_model
            elif args.model_type in {"diffucoder", "diffucoder_parallel"}:
                base_model = model_cfg.get("base_model", "")
                lora_path = model_cfg.get("lora_path", "")
                if not base_model or not lora_path:
                    raise ValueError(
                        "DiffuCoder requires both base_model and lora_path. "
                        "Provide them via --model_path 'base,lora' or set the "
                        "DIFFUCODER_BASE_MODEL and DIFFUCODER_LORA_PATH environment variables."
                    )
                model_path = f"{base_model},{lora_path}"
            else:
                raise ValueError(
                    f"Unsupported configuration dictionary for model type: {args.model_type}"
                )
        else:
            model_path = model_cfg
    print(f"Loading model from {model_path}")

    print(f"Running model={args.model_type}, size={args.model_size}")
    print(f"\tLoad from `{model_path}`")

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model

    using_parallel = (
        args.model_type == "diffucoder_parallel"
        or ("parallel" in args.model_size.lower())
    )
    parallel_kwargs = {}
    if using_parallel:
        parallel_kwargs = {
            "branching_factor": args.parallel_branching_factor,
            "branch_topp": args.parallel_branch_topp,
            "selection_conf_alpha": args.parallel_selection_alpha,
            "branch_verification_mode": not args.parallel_disable_verification,
            "base_branch_competition": not args.parallel_disable_base_competition,
            "verification_force_base_winner": args.parallel_force_base_winner,
            "use_uncertainty_logic": not args.parallel_disable_uncertainty,
            "top_p": args.parallel_top_p,
            "top_k": args.parallel_top_k,
        }
    basic_kwargs = {}
    if args.model_type == "diffucoder_basic":
        basic_kwargs = {
            "token_per_step": args.basic_token_per_step,
        }

    model = make_model(
        model_type=args.model_type,
        model_size=args.model_size,
        model_path=model_path,
        batch_size=args.bs,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        dataset=args.dataset,
        tensor_parallel_size=args.tensor_parallel_size,
        device=args.device,
        # [NEW] Pass new arguments to model constructor (will be caught by **kwargs)
        block_size=args.block_size,
        block_add_threshold=args.block_add_threshold,
        skip_threshold=args.skip_threshold,
        decoded_token_threshold=args.decoded_token_threshold,
        **basic_kwargs,
        **parallel_kwargs,
    )

    run_suffix_parts = []
    if using_parallel:
        def fmt_float(val: float) -> str:
            return ("{:.3f}".format(val)).rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")

        run_suffix_parts.extend(
            [
                f"bf{args.parallel_branching_factor}",
                f"topp{fmt_float(args.parallel_branch_topp)}",
                f"alpha{fmt_float(args.parallel_selection_alpha)}",
            ]
        )
        if args.parallel_top_p is not None:
            run_suffix_parts.append(f"tp{fmt_float(args.parallel_top_p)}")
        if args.parallel_top_k is not None:
            run_suffix_parts.append(f"tk{args.parallel_top_k}")
        if args.parallel_disable_verification:
            run_suffix_parts.append("noVer")
        if args.parallel_disable_base_competition:
            run_suffix_parts.append("noBaseComp")
        if args.parallel_force_base_winner:
            run_suffix_parts.append("forceBase")
        if args.parallel_disable_uncertainty:
            run_suffix_parts.append("noUnc")

    if args.run_tag:
        run_suffix_parts.append(args.run_tag)

    run_suffix = f"_{'_'.join(run_suffix_parts)}" if run_suffix_parts else ""

    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model_type
        + f"_{args.model_size}"
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}")
        + run_suffix,
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"Working dir: {workdir}")

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    print(f"Model cls: {model.__class__}")
    print(f"EOS tokens: {model.eos}")
    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
