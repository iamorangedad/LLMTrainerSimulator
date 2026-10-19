"""训练估算引擎。

口径见 docs/技术方案.md 第 6、7 节。build_plan 是纯函数：相同配置得到相同账单。
损失噪声和慢卡只出现在 simulate 里。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Optional

from app.catalog import FRAMEWORKS, GPUS, INTERCONNECTS, MODELS, _pick
from app.i18n import norm_lang, sim, t

OVERHEAD_BYTES = 1.2 * (1024**3)
PCIE_OFFLOAD_GBPS = 12.0
GIB = 1024**3

Disconnect = Callable[[], Awaitable[bool]]


@dataclass(frozen=True)
class TrainConfig:
    nodes: int
    gpus_per_node: int
    gpu: str
    interconnect: str
    framework: str
    zero_stage: int
    tp: int
    pp: int
    model: str
    micro_batch: int
    grad_accum: int
    seq_len: int
    precision: str
    activation_checkpointing: bool
    cpu_offload: bool
    steps: int
    lr: float
    speed: float


def build_plan(cfg: TrainConfig, lang: str = "zh") -> dict:
    lang = norm_lang(lang)
    errors, warnings = _input_errors(cfg, lang)
    if errors:
        return _skeleton(errors, warnings, lang)

    framework = cfg.framework
    tp = cfg.tp if framework == "megatron" else 1
    pp = cfg.pp if framework == "megatron" else 1
    use_offload = bool(cfg.cpu_offload) if framework == "deepspeed" else False
    zero_stage = cfg.zero_stage if framework == "deepspeed" else 0

    if framework != "megatron" and (cfg.tp != 1 or cfg.pp != 1):
        warnings.append(t(
            lang,
            "当前框架只做数据并行，张量并行和流水线并行按 1 估算。",
            "This framework only does data parallelism. Tensor and pipeline parallelism are estimated as 1.",
        ))
    if framework != "deepspeed" and cfg.cpu_offload:
        warnings.append(t(
            lang,
            "CPU offload 只在 DeepSpeed 下生效，本次按放在 GPU 上估算。",
            "CPU offload applies only to DeepSpeed. This estimate keeps the optimizer on the GPU.",
        ))

    model = MODELS[cfg.model]
    gpu = GPUS[cfg.gpu]
    link = INTERCONNECTS[cfg.interconnect]
    world = cfg.nodes * cfg.gpus_per_node

    if world % (tp * pp) != 0:
        errors.append(t(
            lang,
            f"进程数 {world} 不能被张量并行 {tp} × 流水线并行 {pp} 整除，排不出数据并行组。",
            f"World size {world} is not divisible by tensor parallel {tp} × pipeline parallel {pp}, so there is no data-parallel group.",
        ))
    if model["heads"] % tp != 0:
        errors.append(t(
            lang,
            f"注意力头数 {model['heads']} 不能被张量并行度 {tp} 整除。",
            f"Attention heads {model['heads']} are not divisible by tensor parallel degree {tp}.",
        ))
    if model["layers"] % pp != 0:
        errors.append(t(
            lang,
            f"层数 {model['layers']} 不能被流水线并行度 {pp} 整除。",
            f"Layer count {model['layers']} is not divisible by pipeline parallel degree {pp}.",
        ))
    if framework == "deepspeed" and cfg.zero_stage not in (1, 2, 3):
        errors.append(t(lang, "DeepSpeed 的 ZeRO 阶段需要是 1、2 或 3。", "DeepSpeed ZeRO stage must be 1, 2, or 3."))
    if errors:
        return _skeleton(errors, warnings, lang)

    dp = world // (tp * pp)
    layout = _layout(cfg.nodes, cfg.gpus_per_node, tp, pp, dp)
    tp_cross = _crosses(layout, "tp")
    pp_cross = _crosses(layout, "pp")
    dp_cross = _crosses(layout, "dp")

    if tp > 1 and tp_cross:
        warnings.append(t(
            lang,
            "张量并行组跨出了节点，层内通信按节点间网络估算。",
            "The tensor-parallel group crosses nodes, so intra-layer communication uses the inter-node network.",
        ))

    b_param, b_grad, b_opt = _bytes_per_param(cfg.precision)
    param_frac, grad_frac, opt_frac = _shard_fractions(framework, zero_stage, dp)
    local_params = model["params"] / (tp * pp)
    param_bytes = local_params * b_param * param_frac
    grad_bytes = local_params * b_grad * grad_frac
    opt_bytes = local_params * b_opt * opt_frac
    opt_gpu_bytes = 0.0 if use_offload else opt_bytes
    act_bytes = _activation_bytes(cfg, model, pp, tp if framework == "megatron" else 1)
    transient = _transient_bytes(cfg, model, framework, zero_stage, dp, b_param)
    total_bytes = param_bytes + grad_bytes + opt_gpu_bytes + act_bytes + transient + OVERHEAD_BYTES
    capacity = gpu["mem_gb"] * GIB
    fits = total_bytes <= capacity
    if fits and total_bytes >= 0.92 * capacity:
        warnings.append(t(
            lang,
            "峰值显存已到容量的 92% 以上，分配器碎片可能让真实训练失败。",
            "Peak memory is at least 92% of capacity. Allocator fragmentation can still fail a real run.",
        ))
    if not fits:
        warnings.append(t(
            lang,
            "峰值显存超过这张 GPU 的容量，播放会在分配阶段结束。",
            "Peak memory exceeds this GPU. Playback stops at allocation.",
        ))

    act_dtype = 4 if cfg.precision == "fp32" else 2
    k_act = 2.2 if cfg.activation_checkpointing else 14.0
    full_act = cfg.micro_batch * cfg.seq_len * model["hidden"] * model["layers"] * act_dtype * k_act
    single_static = model["params"] * (b_param + b_grad + (0 if use_offload else b_opt))
    single_gpu_fits = single_static + full_act + OVERHEAD_BYTES <= capacity

    intra = {"gbps": gpu["intra_gbps"], "lat": gpu["intra_lat_us"] * 1e-6, "name": gpu["intra_name"]}
    inter = {"gbps": link["gbps"], "lat": link["lat_us"] * 1e-6, "name": link["name"]}

    replicated_grad = local_params * b_grad
    replicated_param = local_params * b_param
    comm_s = 0.0
    cross_node_comm = False

    if dp > 1 and (framework == "ddp" or framework == "megatron" or (framework == "deepspeed" and zero_stage == 1)):
        comm_s += _slowest(layout, "dp", replicated_grad, intra, inter, "allreduce")
        cross_node_comm = cross_node_comm or dp_cross
    elif dp > 1 and framework == "deepspeed" and zero_stage == 2:
        comm_s += _slowest(layout, "dp", replicated_grad, intra, inter, "reduce_scatter")
        cross_node_comm = cross_node_comm or dp_cross
    elif dp > 1 and (framework == "fsdp" or (framework == "deepspeed" and zero_stage == 3)):
        one = (
            _slowest(layout, "dp", replicated_param, intra, inter, "allgather") * 2
            + _slowest(layout, "dp", replicated_grad, intra, inter, "reduce_scatter")
        )
        comm_s += one * cfg.grad_accum
        cross_node_comm = cross_node_comm or dp_cross

    layers_local = model["layers"] // pp
    if framework == "megatron" and tp > 1:
        msg = cfg.micro_batch * cfg.seq_len * model["hidden"] * act_dtype
        comm_s += _slowest(layout, "tp", msg, intra, inter, "allreduce") * layers_local * 4 * cfg.grad_accum
        cross_node_comm = cross_node_comm or tp_cross

    if pp > 1:
        act_msg = cfg.micro_batch * cfg.seq_len * model["hidden"] * act_dtype
        bus = inter if pp_cross else intra
        p2p = act_msg / (bus["gbps"] * 1e9) + bus["lat"]
        # 气泡已经算进计算时间，点对点只保留一部分。
        comm_s += p2p * cfg.grad_accum * 0.5
        cross_node_comm = cross_node_comm or pp_cross

    eff = 0.28 if model["params"] < 2e9 else 0.42
    peak = gpu["tflops_fp32"] if cfg.precision == "fp32" else gpu["tflops_bf16"]
    tokens_per_gpu = cfg.micro_batch * cfg.grad_accum * cfg.seq_len
    pure_compute = (6 * model["params"] * tokens_per_gpu / (tp * pp)) / (peak * 1e12 * eff)
    if pp == 1:
        bubble = 0.0
        bubble_mult = 1.0
    else:
        bubble = (pp - 1) / (cfg.grad_accum + pp - 1)
        bubble_mult = (cfg.grad_accum + pp - 1) / cfg.grad_accum
    compute_s = pure_compute * bubble_mult

    overlap = _overlap(framework, zero_stage)
    exposed = max(0.0, comm_s - overlap * pure_compute)

    touch = param_bytes + grad_bytes + opt_gpu_bytes
    opt_s = touch / (gpu["membw_gbps"] * 1e9 * 0.25)
    if use_offload:
        opt_s += opt_bytes / (PCIE_OFFLOAD_GBPS * 1e9)

    data_s = 0.025 if cfg.nodes > 1 else 0.015
    step_s = data_s + compute_s + exposed + opt_s

    global_samples = cfg.micro_batch * cfg.grad_accum * dp
    global_tokens = global_samples * cfg.seq_len
    single_compute = (6 * model["params"] * global_tokens) / (peak * 1e12 * eff)
    speedup = single_compute / step_s
    tokens_per_s = global_tokens / step_s
    mfu = (6 * model["params"] * global_tokens / step_s) / (peak * 1e12 * world)

    if pp > 1 and cfg.grad_accum < pp:
        warnings.append(t(
            lang,
            "微批次数少于流水线阶段数，气泡会占掉大部分时间。",
            "There are fewer microbatches than pipeline stages, so the bubble takes most of the step.",
        ))
    if exposed > pure_compute and pure_compute > 0:
        warnings.append(t(lang, "暴露通信时间长于计算时间。", "Exposed communication is longer than compute."))

    link_name = _pick(link, lang)
    if cross_node_comm and cfg.nodes > 1:
        comm_bus = t(
            lang,
            f"节点间 {link['name']}，分层集合通信：节点内归约，节点间同步，节点内再分发",
            f"Inter-node {link_name}, hierarchical collective: reduce inside the node, sync across nodes, then distribute inside the node",
        )
    elif comm_s > 0:
        comm_bus = t(lang, f"节点内 {gpu['intra_name']}", f"Intra-node {gpu['intra_name']}")
    else:
        comm_bus = t(lang, "这一步没有卡间通信", "This step has no GPU-to-GPU communication")

    memory = {
        "param_gb": _gb(param_bytes),
        "grad_gb": _gb(grad_bytes),
        "optim_gb": _gb(opt_gpu_bytes),
        "optim_offload_gb": _gb(opt_bytes if use_offload else 0),
        "activation_gb": _gb(act_bytes),
        "transient_gb": _gb(transient),
        "overhead_gb": _gb(OVERHEAD_BYTES),
        "total_gb": _gb(total_bytes),
        "gpu_gb": gpu["mem_gb"],
        "fits": fits,
    }
    parallel = {"world": world, "dp": dp, "tp": tp, "pp": pp, "zero_stage": zero_stage}
    timing = {
        "data_ms": _ms(data_s),
        "compute_ms": _ms(compute_s),
        "pure_compute_ms": _ms(pure_compute),
        "comm_ms": _ms(comm_s),
        "exposed_comm_ms": _ms(exposed),
        "optim_ms": _ms(opt_s),
        "step_ms": _ms(step_s),
        "bubble_frac": round(bubble, 4),
        "tokens_per_s": round(tokens_per_s, 1),
        "mfu": round(mfu, 4),
        "speedup": round(speedup, 3),
        "single_gpu_fits": single_gpu_fits,
        "comm_bus": comm_bus,
    }
    batch = {
        "micro": cfg.micro_batch,
        "grad_accum": cfg.grad_accum,
        "global_samples": global_samples,
        "global_tokens": global_tokens,
    }
    cluster = _cluster(layout, cfg.nodes)
    precision_name = {
        "bf16": t(lang, "BF16 混合精度", "BF16 mixed precision"),
        "fp16": t(lang, "FP16 混合精度", "FP16 mixed precision"),
        "fp32": "FP32",
    }[cfg.precision]
    labels = {
        "framework": FRAMEWORKS[framework]["name"],
        "framework_id": framework,
        "model": _pick(model, lang),
        "gpu": gpu["name"],
        "gpu_id": cfg.gpu,
        "interconnect": link_name,
        "link_id": cfg.interconnect,
        "intra": gpu["intra_name"],
        "precision": precision_name,
    }
    owned_params = local_params * param_frac
    notes = _notes(
        cfg,
        framework,
        zero_stage,
        use_offload,
        tp,
        pp,
        dp,
        model,
        gpu,
        link,
        memory,
        timing,
        batch,
        tp_cross,
        cluster,
        fits,
        lang,
    )
    formulas = [
        t(
            lang,
            "参数、梯度和优化器：混合精度按每参数 16 字节（2+2+12），FP32 按 4+4+8。激活只随微批次和序列长度增长，重计算把激活系数从 14 降到 2.2。",
            "Parameters, gradients, and optimizer state: mixed precision uses 16 bytes per parameter (2+2+12), FP32 uses 4+4+8. Activation grows only with micro-batch and sequence length. Recomputation lowers the activation factor from 14 to 2.2.",
        ),
        t(
            lang,
            "计算量按每 token 6 倍参数量估算。20 亿参数以下有效算力取峰值的 28%，否则取 42%。流水线气泡按 (阶段数−1)/(微批次数+阶段数−1) 拉长计算。",
            "Compute is estimated as 6 times the parameter count per token. Effective throughput is 28% of peak below 2 billion parameters, otherwise 42%. The pipeline bubble stretches compute by (stages−1)/(microbatches+stages−1).",
        ),
        t(
            lang,
            f"通信按 ring 公式；跨节点时分层计算。本次可被计算掩盖的比例是 {overlap:.2f}。暴露通信 = max(0, 通信时间 − 该比例 × 纯计算时间)。系数为教学设定，不是实测跑分。",
            f"Communication uses the ring formulas, and is hierarchical across nodes. The share hidden behind compute this time is {overlap:.2f}. Exposed communication = max(0, communication − that share × pure compute). The factors are teaching settings, not a benchmark.",
        ),
    ]
    return {
        "lang": lang,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "cluster": cluster,
        "memory": memory,
        "parallel": parallel,
        "batch": batch,
        "timing": timing,
        "formulas": formulas,
        "labels": labels,
        "owned_params": owned_params,
        "full_params": model["params"],
        "layers_local": layers_local,
    }


def startup_lines(plan: dict) -> list[tuple[str, str]]:
    """启动日志，顺序对应技术方案第 7 节。"""
    lang = norm_lang(plan.get("lang"))
    tag = sim(lang)
    if plan["errors"]:
        return [("error", f"{tag} {err}") for err in plan["errors"]]

    p = plan["parallel"]
    b = plan["batch"]
    m = plan["memory"]
    labels = plan["labels"]
    cluster = plan["cluster"]["nodes"]
    master = cluster[0]["host"]
    lines: list[tuple[str, str]] = []
    lines.append((
        "info",
        t(
            lang,
            (
                f"{tag} {len(cluster)} 节点 × 每节点 {len(cluster[0]['gpus'])} 张 {labels['gpu']}，"
                f"框架 {labels['framework']}，精度 {labels['precision']}，"
                f"world={p['world']}，dp={p['dp']}，tp={p['tp']}，pp={p['pp']}，"
                f"全局样本数 {b['global_samples']}，全局 token {b['global_tokens']}。"
            ),
            (
                f"{tag} {len(cluster)} nodes × {len(cluster[0]['gpus'])} {labels['gpu']} per node, "
                f"framework {labels['framework']}, precision {labels['precision']}, "
                f"world={p['world']}, dp={p['dp']}, tp={p['tp']}, pp={p['pp']}, "
                f"global samples {b['global_samples']}, global tokens {b['global_tokens']}."
            ),
        ),
    ))
    lines.extend(_launch_lines(plan, master))
    world = p["world"]
    for node in cluster:
        ranks = [gpu["rank"] for gpu in node["gpus"]]
        rank_text = str(ranks[0]) if ranks[0] == ranks[-1] else f"{ranks[0]}-{ranks[-1]}"
        lines.append((
            "info",
            t(
                lang,
                (
                    f"{tag}[node{node['id']} {node['host']}] ranks {rank_text} 加入进程组 "
                    f"backend=nccl master={master}:29500 world_size={world}"
                ),
                (
                    f"{tag}[node{node['id']} {node['host']}] ranks {rank_text} joined the process group "
                    f"backend=nccl master={master}:29500 world_size={world}"
                ),
            ),
        ))
    lines.extend(_fabric_lines(plan))
    lines.append((
        "info",
        t(
            lang,
            f"{tag} 模型参数量 {plan['full_params']:.3e}；本卡持有参数 {plan['owned_params']:.3e}。",
            f"{tag} Model parameters {plan['full_params']:.3e}; this GPU owns {plan['owned_params']:.3e}.",
        ),
    ))
    lines.append((
        "info",
        t(
            lang,
            (
                f"{tag} 显存估算 "
                f"参数 {m['param_gb']:.2f} GB | 梯度 {m['grad_gb']:.2f} GB | "
                f"优化器 {m['optim_gb']:.2f} GB | 激活 {m['activation_gb']:.2f} GB | "
                f"临时拼参 {m['transient_gb']:.2f} GB | 开销 {m['overhead_gb']:.2f} GB | "
                f"合计 {m['total_gb']:.2f} / {m['gpu_gb']:.0f} GB"
            ),
            (
                f"{tag} Memory estimate "
                f"params {m['param_gb']:.2f} GB | grads {m['grad_gb']:.2f} GB | "
                f"optimizer {m['optim_gb']:.2f} GB | activation {m['activation_gb']:.2f} GB | "
                f"gather {m['transient_gb']:.2f} GB | overhead {m['overhead_gb']:.2f} GB | "
                f"total {m['total_gb']:.2f} / {m['gpu_gb']:.0f} GB"
            ),
        ),
    ))
    if m["optim_offload_gb"] > 0:
        lines.append(("info", t(
            lang,
            f"{tag} 另有优化器状态 {m['optim_offload_gb']:.2f} GB 放在 CPU。",
            f"{tag} Another {m['optim_offload_gb']:.2f} GB of optimizer state sits on the CPU.",
        )))
    note_tag = t(lang, f"{tag}[注意]", f"{tag}[note]")
    joiner = "。" if lang == "zh" else ". "
    for note in plan["notes"]:
        level = "warn" if note["level"] == "warn" else "info"
        lines.append((level, f"{note_tag} {note['title']}{joiner}{note['body']}"))
    if not m["fits"]:
        lines.append((
            "error",
            t(
                lang,
                (
                    f"{tag} 分配失败：峰值 {m['total_gb']:.2f} GB，"
                    f"超过 {labels['gpu']} 的 {m['gpu_gb']:.0f} GB。训练步未开始。"
                ),
                (
                    f"{tag} Allocation failed: peak {m['total_gb']:.2f} GB "
                    f"exceeds the {m['gpu_gb']:.0f} GB on {labels['gpu']}. Training steps did not start."
                ),
            ),
        ))
    else:
        lines.append(("info", t(
            lang,
            f"{tag} 显存账单落在容量以内，进入训练循环。",
            f"{tag} The memory bill fits, entering the training loop.",
        )))
    return lines


def checkpoint_line(plan: dict) -> str:
    lang = norm_lang(plan.get("lang"))
    p = plan["parallel"]
    fw = plan["labels"]["framework"]
    dp = p["dp"]
    if p["zero_stage"] == 3 or fw == "PyTorch FSDP":
        how = t(
            lang,
            f"参数、梯度和优化器状态沿 dp={dp} 切分后写出",
            f"parameters, gradients, and optimizer state are written sharded across dp={dp}",
        )
    elif p["zero_stage"] == 2:
        how = t(
            lang,
            f"优化器状态和梯度沿 dp={dp} 切分后写出",
            f"optimizer state and gradients are written sharded across dp={dp}",
        )
    elif p["zero_stage"] == 1 or fw == "Megatron-LM":
        how = t(
            lang,
            f"优化器状态沿 dp={dp} 切分后写出",
            f"optimizer state is written sharded across dp={dp}",
        )
    elif dp > 1:
        how = t(lang, "每张卡各写一份完整优化器状态", "each GPU writes a full copy of the optimizer state")
    else:
        how = t(lang, "单卡写出完整优化器状态", "the single GPU writes the full optimizer state")
    return t(
        lang,
        f"{sim(lang)}[checkpoint] {how}。这里不模拟磁盘时间。",
        f"{sim(lang)}[checkpoint] {how}. Disk time is not simulated.",
    )


def warmup_steps(steps: int) -> int:
    if steps <= 1:
        return 1
    return max(1, min(10, steps // 5))


def lr_at(step: int, steps: int, base: float) -> float:
    warm = warmup_steps(steps)
    if step <= warm:
        return base * step / warm
    return base


def spike_step(cfg: TrainConfig) -> Optional[int]:
    if cfg.nodes > 1 and cfg.steps >= 8:
        return cfg.steps // 2
    return None


def loss_value(step: int, steps: int, rng: random.Random, spiked: Optional[int]) -> float:
    value = 1.85 + (4.6 - 1.85) * math.exp(-3.0 * step / max(steps, 1))
    value += rng.gauss(0, 0.02)
    if spiked is not None and step == spiked:
        value += 0.35
    return max(0.05, value)


def config_seed(cfg: TrainConfig) -> int:
    payload = asdict(cfg)
    payload.pop("speed", None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def loss_series(cfg: TrainConfig) -> list[float]:
    rng = random.Random(config_seed(cfg))
    spiked = spike_step(cfg)
    return [loss_value(step, cfg.steps, rng, spiked) for step in range(1, cfg.steps + 1)]


async def simulate(cfg: TrainConfig, is_disconnected: Optional[Disconnect] = None, lang: str = "zh"):
    plan = build_plan(cfg, lang)
    yield {"kind": "status", "status": "init"}
    yield {"kind": "plan", "plan": plan}
    if plan["errors"]:
        for _level, text in startup_lines(plan):
            yield {"kind": "log", "level": "error", "text": text}
        yield {"kind": "status", "status": "error"}
        return

    speed = max(1.0, float(cfg.speed))
    for level, text in startup_lines(plan):
        yield {"kind": "log", "level": level, "text": text}
        if not await _pause(0.16 / speed, is_disconnected):
            return
    if not plan["memory"]["fits"]:
        yield {"kind": "status", "status": "oom"}
        return

    yield {"kind": "status", "status": "running"}
    rng = random.Random(config_seed(cfg))
    spiked = spike_step(cfg)
    slow_at = max(2, cfg.steps // 3) if cfg.nodes > 1 and cfg.steps >= 4 else None
    ckpt_at = cfg.steps // 2 if cfg.steps >= 8 else None
    timing = plan["timing"]
    played_steps: list[float] = []
    last_node = plan["cluster"]["nodes"][-1]
    last_rank = last_node["gpus"][-1]["rank"]

    for step in range(1, cfg.steps + 1):
        phase_sleep = (0.42 if step == 1 else 0.16) / speed
        phases = ("data", "forward", "backward", "comm", "optim")
        for phase in phases:
            event = {"kind": "phase", "phase": phase, "step": step}
            if phase == "comm":
                event["bus"] = timing["comm_bus"]
            yield event
            if step == 1:
                yield {"kind": "log", "level": "info", "text": _first_step_line(plan, phase, cfg)}
            if not await _pause(phase_sleep, is_disconnected):
                return

        loss = loss_value(step, cfg.steps, rng, spiked)
        current_lr = lr_at(step, cfg.steps, cfg.lr)
        factor = 1.0
        if slow_at is not None and step == slow_at:
            factor = 1.25
            yield {
                "kind": "note",
                "note": {
                    "id": "straggler",
                    "level": "warn",
                    "title": t(plan.get("lang"), "慢卡", "Straggler"),
                    "body": t(
                        plan.get("lang"),
                        (
                            f"node{last_node['id']} 的 rank {last_rank} 这一步更慢。"
                            "集合通信要等组里最慢的 rank，整步时间一起被拉长。"
                        ),
                        (
                            f"Rank {last_rank} on node{last_node['id']} is slower on this step. "
                            "The collective waits for the slowest rank, so the whole step gets longer."
                        ),
                    ),
                },
            }
            yield {
                "kind": "log",
                "level": "warn",
                "text": t(
                    plan.get("lang"),
                    (
                        f"[模拟][node{last_node['id']} rank{last_rank}] "
                        "本步集体操作等待最慢的 rank，步时额外加上 25% 的等待。"
                    ),
                    (
                        f"[sim][node{last_node['id']} rank{last_rank}] "
                        "This collective waits for the slowest rank, adding 25% to the step."
                    ),
                ),
            }
        if spiked is not None and step == spiked:
            yield {
                "kind": "note",
                "note": {
                    "id": "loss-spike",
                    "level": "info",
                    "title": t(plan.get("lang"), "损失跳动", "Loss spike"),
                    "body": t(
                        plan.get("lang"),
                        "这一步的损失往上跳了一截。学习率、数据顺序或同步都可能造成跳动。模拟器只把它做成可观察的现象，不判断原因。",
                        "The loss jumps on this step. Learning rate, data order, or synchronization can all cause that. The simulator only makes the jump visible; it does not diagnose a cause.",
                    ),
                },
            }

        step_ms = timing["step_ms"] * factor
        exposed_ms = timing["exposed_comm_ms"] + timing["step_ms"] * (factor - 1.0)
        played_steps.append(step_ms)
        metrics = {
            "kind": "metrics",
            "step": step,
            "steps": cfg.steps,
            "loss": round(loss, 4),
            "lr": current_lr,
            "tokens_per_s": round(timing["tokens_per_s"] / factor, 1),
            "mem_gb": plan["memory"]["total_gb"],
            "mfu": round(timing["mfu"] / factor, 4),
            "data_ms": timing["data_ms"],
            "compute_ms": timing["compute_ms"],
            "exposed_comm_ms": round(exposed_ms, 2),
            "optim_ms": timing["optim_ms"],
            "step_ms": round(step_ms, 2),
            "bubble_frac": timing["bubble_frac"],
        }
        yield {"kind": "log", "level": "info", "text": _step_summary(metrics, plan.get("lang"))}
        yield metrics

        if ckpt_at is not None and step == ckpt_at:
            yield {"kind": "phase", "phase": "checkpoint", "step": step}
            yield {"kind": "log", "level": "info", "text": checkpoint_line(plan)}
            if not await _pause(0.2 / speed, is_disconnected):
                return

        if step == cfg.steps:
            eval_loss = loss + 0.15
            yield {
                "kind": "log",
                "level": "info",
                "text": t(
                    plan.get("lang"),
                    f"[模拟][eval] 验证损失 {eval_loss:.3f}。这是教学曲线，不是某个数据集上的成绩。",
                    f"[sim][eval] Validation loss {eval_loss:.3f}. This is a teaching curve, not a score on a dataset.",
                ),
            }

    avg_ms = sum(played_steps) / len(played_steps)
    avg_s = avg_ms / 1000
    global_tokens = plan["batch"]["global_tokens"]
    # 用播放中的平均步时回算吞吐，慢卡那一步会计入平均。
    base_step_s = timing["step_ms"] / 1000
    scale = base_step_s / avg_s if avg_s else 1.0
    yield {
        "kind": "log",
        "level": "info",
        "text": t(
            plan.get("lang"),
            (
                f"[模拟] 播放结束。本次平均步时 {avg_ms:.0f} ms（预览步时 {timing['step_ms']:.0f} ms），"
                f"吞吐 {timing['tokens_per_s'] * scale:,.0f} tok/s，"
                f"MFU {timing['mfu'] * scale * 100:.1f}%，"
                f"相对单卡计算的理论加速比 {timing['speedup'] * scale:.2f}。"
                "以上为教学估算。"
            ),
            (
                f"[sim] Playback finished. Mean step time {avg_ms:.0f} ms (preview step {timing['step_ms']:.0f} ms), "
                f"throughput {timing['tokens_per_s'] * scale:,.0f} tok/s, "
                f"MFU {timing['mfu'] * scale * 100:.1f}%, "
                f"speedup versus single-GPU compute {timing['speedup'] * scale:.2f}. "
                "These are teaching estimates."
            ),
        ),
    }
    yield {"kind": "status", "status": "done"}


def _input_errors(cfg: TrainConfig, lang: str = "zh") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if cfg.gpu not in GPUS:
        errors.append(t(lang, f"未知 GPU：{cfg.gpu}", f"Unknown GPU: {cfg.gpu}"))
    if cfg.interconnect not in INTERCONNECTS:
        errors.append(t(lang, f"未知互联：{cfg.interconnect}", f"Unknown interconnect: {cfg.interconnect}"))
    if cfg.model not in MODELS:
        errors.append(t(lang, f"未知模型：{cfg.model}", f"Unknown model: {cfg.model}"))
    if cfg.framework not in FRAMEWORKS:
        errors.append(t(lang, f"未知框架：{cfg.framework}", f"Unknown framework: {cfg.framework}"))
    if cfg.precision not in ("bf16", "fp16", "fp32"):
        errors.append(t(lang, f"未知精度：{cfg.precision}", f"Unknown precision: {cfg.precision}"))
    if not 1 <= cfg.nodes <= 32:
        errors.append(t(lang, "节点数需要在 1 到 32 之间。", "Node count must be from 1 to 32."))
    if not 1 <= cfg.gpus_per_node <= 8:
        errors.append(t(lang, "每节点 GPU 数需要在 1 到 8 之间。", "GPUs per node must be from 1 to 8."))
    if not 1 <= cfg.steps <= 300:
        errors.append(t(lang, "步数需要在 1 到 300 之间。", "Step count must be from 1 to 300."))
    if not 128 <= cfg.seq_len <= 8192:
        errors.append(t(lang, "序列长度需要在 128 到 8192 之间。", "Sequence length must be from 128 to 8192."))
    if not 1 <= cfg.micro_batch <= 32:
        errors.append(t(lang, "微批次需要在 1 到 32 之间。", "Micro-batch must be from 1 to 32."))
    if not 1 <= cfg.grad_accum <= 64:
        errors.append(t(lang, "梯度累积需要在 1 到 64 之间。", "Gradient accumulation must be from 1 to 64."))
    if cfg.tp < 1 or cfg.pp < 1:
        errors.append(t(lang, "张量并行度和流水线并行度至少为 1。", "Tensor and pipeline parallel degrees must be at least 1."))
    if not 0 < cfg.lr <= 1:
        errors.append(t(lang, "学习率需要大于 0 且不超过 1。", "Learning rate must be greater than 0 and at most 1."))
    if not 1 <= cfg.speed <= 8:
        errors.append(t(lang, "播放速度需要在 1 到 8 之间。", "Playback speed must be from 1 to 8."))
    return errors, []


def _skeleton(errors: list[str], warnings: list[str], lang: str = "zh") -> dict:
    zero = {
        "data_ms": 0.0,
        "compute_ms": 0.0,
        "pure_compute_ms": 0.0,
        "comm_ms": 0.0,
        "exposed_comm_ms": 0.0,
        "optim_ms": 0.0,
        "step_ms": 0.0,
        "bubble_frac": 0.0,
        "tokens_per_s": 0.0,
        "mfu": 0.0,
        "speedup": 0.0,
        "single_gpu_fits": False,
        "comm_bus": "",
    }
    return {
        "lang": norm_lang(lang),
        "errors": errors,
        "warnings": warnings,
        "notes": [],
        "cluster": {"nodes": []},
        "memory": {
            "param_gb": 0.0,
            "grad_gb": 0.0,
            "optim_gb": 0.0,
            "optim_offload_gb": 0.0,
            "activation_gb": 0.0,
            "transient_gb": 0.0,
            "overhead_gb": 0.0,
            "total_gb": 0.0,
            "gpu_gb": 0.0,
            "fits": False,
        },
        "parallel": {"world": 0, "dp": 0, "tp": 0, "pp": 0, "zero_stage": 0},
        "batch": {"micro": 0, "grad_accum": 0, "global_samples": 0, "global_tokens": 0},
        "timing": zero,
        "formulas": [],
        "labels": {},
        "owned_params": 0.0,
        "full_params": 0.0,
        "layers_local": 0,
    }


def _bytes_per_param(precision: str) -> tuple[int, int, int]:
    if precision == "fp32":
        return 4, 4, 8
    return 2, 2, 12


def _shard_fractions(framework: str, zero_stage: int, dp: int) -> tuple[float, float, float]:
    if framework == "ddp":
        return 1.0, 1.0, 1.0
    if framework == "fsdp":
        return 1 / dp, 1 / dp, 1 / dp
    if framework == "deepspeed":
        if zero_stage == 1:
            return 1.0, 1.0, 1 / dp
        if zero_stage == 2:
            return 1.0, 1 / dp, 1 / dp
        return 1 / dp, 1 / dp, 1 / dp
    # Megatron：权重已按 tp、pp 切开，优化器状态再按 dp 切开。
    return 1.0, 1.0, 1 / dp


def _activation_bytes(cfg: TrainConfig, model: dict, pp: int, tp_div: int) -> float:
    dtype = 4 if cfg.precision == "fp32" else 2
    k = 2.2 if cfg.activation_checkpointing else 14.0
    layers_local = model["layers"] / pp
    return cfg.micro_batch * cfg.seq_len * model["hidden"] * layers_local * dtype * k / tp_div


def _transient_bytes(cfg: TrainConfig, model: dict, framework: str, zero_stage: int, dp: int, b_param: int) -> float:
    sharded = framework == "fsdp" or (framework == "deepspeed" and zero_stage == 3)
    if not sharded or dp <= 1:
        return 0.0
    full_layer = model["params"] / model["layers"] * b_param
    return full_layer * (1 - 1 / dp)


def _overlap(framework: str, zero_stage: int) -> float:
    if framework == "ddp":
        return 0.70
    if framework == "deepspeed" and zero_stage in (1, 2):
        return 0.45
    if framework == "fsdp" or (framework == "deepspeed" and zero_stage == 3):
        return 0.30
    if framework == "megatron":
        return 0.35
    return 0.0


def _layout(nodes: int, gpus: int, tp: int, pp: int, dp: int) -> list[dict]:
    rows = []
    for dp_rank in range(dp):
        for pp_rank in range(pp):
            for tp_rank in range(tp):
                rank = (dp_rank * pp + pp_rank) * tp + tp_rank
                rows.append({
                    "rank": rank,
                    "node": rank // gpus,
                    "local": rank % gpus,
                    "tp": tp_rank,
                    "pp": pp_rank,
                    "dp": dp_rank,
                })
    rows.sort(key=lambda item: item["rank"])
    return rows


def _cluster(layout: list[dict], nodes: int) -> dict:
    grouped = []
    for node_id in range(nodes):
        grouped.append({
            "id": node_id,
            "host": f"10.0.0.{11 + node_id}",
            "gpus": [gpu for gpu in layout if gpu["node"] == node_id],
        })
    return {"nodes": grouped}


def _crosses(layout: list[dict], dim: str) -> bool:
    groups: dict[tuple, set[int]] = {}
    for gpu in layout:
        if dim == "tp":
            key = (gpu["dp"], gpu["pp"])
        elif dim == "pp":
            key = (gpu["dp"], gpu["tp"])
        else:
            key = (gpu["tp"], gpu["pp"])
        groups.setdefault(key, set()).add(gpu["node"])
    return any(len(nodes) > 1 for nodes in groups.values())


def _groups(layout: list[dict], dim: str) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for gpu in layout:
        if dim == "tp":
            key = (gpu["dp"], gpu["pp"])
        elif dim == "pp":
            key = (gpu["dp"], gpu["tp"])
        else:
            key = (gpu["tp"], gpu["pp"])
        groups.setdefault(key, []).append(gpu["rank"])
    return groups


def _ring(nbytes: float, participants: int, gbps: float, lat: float, kind: str) -> float:
    if participants <= 1 or nbytes <= 0:
        return 0.0
    if kind == "allreduce":
        factor = 2 * (participants - 1) / participants
        latency = 2 * (participants - 1) * lat
    else:
        factor = (participants - 1) / participants
        latency = (participants - 1) * lat
    return factor * nbytes / (gbps * 1e9) + latency


def _collective(nbytes: float, ranks: list[int], layout: list[dict], intra: dict, inter: dict, kind: str) -> float:
    if len(ranks) <= 1 or nbytes <= 0:
        return 0.0
    by_node: dict[int, int] = {}
    for rank in ranks:
        node = layout[rank]["node"]
        by_node[node] = by_node.get(node, 0) + 1
    if len(by_node) == 1:
        return _ring(nbytes, len(ranks), intra["gbps"], intra["lat"], kind)
    local_n = max(by_node.values())
    node_n = len(by_node)
    if kind == "allreduce":
        shard = nbytes / local_n
        return (
            _ring(nbytes, local_n, intra["gbps"], intra["lat"], "reduce_scatter")
            + _ring(shard, node_n, inter["gbps"], inter["lat"], "allreduce")
            + _ring(nbytes, local_n, intra["gbps"], intra["lat"], "allgather")
        )
    if kind == "reduce_scatter":
        return (
            _ring(nbytes, local_n, intra["gbps"], intra["lat"], "reduce_scatter")
            + _ring(nbytes / local_n, node_n, inter["gbps"], inter["lat"], "reduce_scatter")
        )
    return (
        _ring(nbytes / node_n, node_n, inter["gbps"], inter["lat"], "allgather")
        + _ring(nbytes, local_n, intra["gbps"], intra["lat"], "allgather")
    )


def _slowest(layout: list[dict], dim: str, nbytes: float, intra: dict, inter: dict, kind: str) -> float:
    times = [
        _collective(nbytes, ranks, layout, intra, inter, kind)
        for ranks in _groups(layout, dim).values()
    ]
    return max(times) if times else 0.0


def _gb(num_bytes: float) -> float:
    return round(num_bytes / GIB, 3)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def _launch_lines(plan: dict, master: str) -> list[tuple[str, str]]:
    lang = norm_lang(plan.get("lang"))
    tag = sim(lang)
    labels = plan["labels"]
    parallel = plan["parallel"]
    nodes = plan["cluster"]["nodes"]
    gpus = len(nodes[0]["gpus"])
    n = len(nodes)
    fw = labels["framework"]
    lines = []
    if fw == "DeepSpeed ZeRO":
        lines.append((
            "info",
            f"{tag} deepspeed --hostfile hostfile --num_nodes {n} --num_gpus {gpus} train.py --deepspeed ds_config.json",
        ))
        lines.append(("info", t(lang, f"{tag} hostfile 内容：", f"{tag} hostfile contents:")))
        for node in nodes:
            lines.append(("info", f"{tag} {node['host']} slots={gpus}"))
        offload = "CPU" if plan["memory"]["optim_offload_gb"] > 0 else "none"
        lines.append((
            "info",
            f"{tag} ds_config zero_optimization.stage={parallel['zero_stage']} offload_optimizer={offload}",
        ))
    else:
        extra = ""
        if fw == "Megatron-LM":
            extra = (
                f" --tensor-model-parallel-size {parallel['tp']}"
                f" --pipeline-model-parallel-size {parallel['pp']}"
            )
        elif fw == "PyTorch FSDP":
            extra = " --fsdp full_shard"
        lines.append((
            "info",
            (
                f"{tag} torchrun --nnodes={n} --nproc_per_node={gpus} "
                f"--node_rank=<0..{n - 1}> --master_addr={master} --master_port=29500 train.py{extra}"
            ),
        ))
    return lines


def _fabric_lines(plan: dict) -> list[tuple[str, str]]:
    lang = norm_lang(plan.get("lang"))
    tag = sim(lang)
    labels = plan["labels"]
    nodes = plan["cluster"]["nodes"]
    gpu = GPUS[labels["gpu_id"]]
    link = INTERCONNECTS[labels["link_id"]]
    lines = [(
        "info",
        t(
            lang,
            (
                f"{tag} 节点内总线 {gpu['intra_name']}，有效带宽 {gpu['intra_gbps']} GB/s，"
                f"时延 {gpu['intra_lat_us']} μs。"
            ),
            (
                f"{tag} Intra-node bus {gpu['intra_name']}, effective bandwidth {gpu['intra_gbps']} GB/s, "
                f"latency {gpu['intra_lat_us']} μs."
            ),
        ),
    )]
    if len(nodes) == 1:
        lines.append(("info", t(
            lang,
            f"{tag} 本次只有一台机器，估算只使用节点内总线。",
            f"{tag} Only one machine is in this run, so the estimate uses the intra-node bus only.",
        )))
    else:
        lines.append((
            "info",
            t(
                lang,
                (
                    f"{tag} 节点间网络 {labels['interconnect']}，有效带宽 {link['gbps']} GB/s，"
                    f"时延 {link['lat_us']} μs。跨节点集合通信按分层方式估算：节点内归约，节点间同步，节点内再分发。"
                ),
                (
                    f"{tag} Inter-node network {labels['interconnect']}, effective bandwidth {link['gbps']} GB/s, "
                    f"latency {link['lat_us']} μs. Collectives that cross nodes are hierarchical: reduce inside the node, sync across nodes, then distribute inside the node."
                ),
            ),
        ))
        lines.append(("info", f"{tag} NCCL_SOCKET_IFNAME=eth0"))
        ib = labels["link_id"].startswith("ib_")
        lines.append(("info", f"{tag} NCCL_IB_DISABLE={0 if ib else 1}"))
    return lines


def _notes(cfg, framework, zero_stage, use_offload, tp, pp, dp, model, gpu, link, memory, timing, batch, tp_cross, cluster, fits, lang="zh") -> list[dict]:
    lang = norm_lang(lang)
    notes = []
    fw_name = FRAMEWORKS[framework]["name"]
    texts = {
        "ddp": t(
            lang,
            "每张卡保存完整的参数、梯度和优化器状态。各卡用不同数据做前向和反向，优化器更新前用 AllReduce 把梯度取平均。",
            "Each GPU keeps full parameters, gradients, and optimizer state. Ranks run different data, then AllReduce the gradients before the optimizer step.",
        ),
        "fsdp": t(
            lang,
            "参数、梯度和优化器状态沿数据并行切分。计算某一层之前先 AllGather 把这一层参数拼回来，反向后用 ReduceScatter 把梯度分回各卡。",
            "Parameters, gradients, and optimizer state are sharded across data-parallel ranks. AllGather rebuilds a layer before it is computed, and ReduceScatter sends gradients back afterward.",
        ),
        "megatron": t(
            lang,
            "张量并行切开每一层里的矩阵，流水线并行把不同层放到不同阶段，数据并行再复制几份。优化器状态沿数据并行切分。",
            "Tensor parallelism splits the matrices inside each layer, pipeline parallelism places layers on different stages, and data parallelism replicates that pair. Optimizer state is sharded across data-parallel ranks.",
        ),
    }
    if framework == "deepspeed":
        body = {
            1: t(lang, "ZeRO-1 把优化器状态按数据并行切开，参数和梯度仍在每张卡上各有一份。", "ZeRO-1 shards optimizer state across data-parallel ranks. Parameters and gradients stay replicated."),
            2: t(lang, "ZeRO-2 在 ZeRO-1 之上把梯度也切开。参数仍完整放在每张卡上。", "ZeRO-2 also shards gradients. Parameters stay replicated on every GPU."),
            3: t(lang, "ZeRO-3 把参数、梯度和优化器状态都切开。算到某一层时再临时把这一层参数拼回来。", "ZeRO-3 shards parameters, gradients, and optimizer state. A layer's parameters are gathered only while that layer runs."),
        }[zero_stage]
    else:
        body = texts[framework]
    notes.append({"id": "framework", "level": "info", "title": fw_name, "body": body})
    notes.append({
        "id": "batch",
        "level": "info",
        "title": t(lang, "全局批次", "Global batch"),
        "body": t(
            lang,
            (
                f"全局样本数 = 微批次 {batch['micro']} × 梯度累积 {batch['grad_accum']} × 数据并行度 {dp} "
                f"= {batch['global_samples']}。再乘序列长度 {cfg.seq_len}，全局 token 数是 {batch['global_tokens']}。"
                "梯度累积增加的是每次优化器更新前看过的样本，单次前向的激活仍按微批次计算。"
            ),
            (
                f"Global samples = micro-batch {batch['micro']} × gradient accumulation {batch['grad_accum']} × data-parallel degree {dp} "
                f"= {batch['global_samples']}. Times sequence length {cfg.seq_len}, that is {batch['global_tokens']} global tokens. "
                "Accumulation increases how many samples are seen before an optimizer step. Activation for one forward still follows the micro-batch."
            ),
        ),
    })
    parts = [
        (
            t(lang, "参数", "Parameters"),
            memory["param_gb"],
            t(lang, "提高切分程度可以减小这一项，例如 FSDP、ZeRO-3，或在 Megatron 里增大张量并行、流水线并行。", "More sharding shrinks this term: FSDP, ZeRO-3, or a larger tensor or pipeline degree in Megatron."),
        ),
        (
            t(lang, "梯度", "Gradients"),
            memory["grad_gb"],
            t(lang, "ZeRO-2 及以上会把梯度切到数据并行的各卡上。", "ZeRO-2 and above shard gradients across the data-parallel ranks."),
        ),
        (
            t(lang, "优化器", "Optimizer"),
            memory["optim_gb"],
            t(lang, "优化器状态通常是大头。ZeRO-1 就开始切它。DeepSpeed 还可以把优化器状态放到 CPU，GPU 显存下降，步时里会多出 PCIe 搬运。", "Optimizer state is usually the largest term. ZeRO-1 starts sharding it. DeepSpeed can also park it on the CPU: GPU memory drops, and the step gains a PCIe transfer."),
        ),
        (
            t(lang, "激活", "Activation"),
            memory["activation_gb"],
            t(lang, "减小微批次或序列长度会直接减小激活。打开激活重计算会把激活系数从 14 降到 2.2。", "A smaller micro-batch or sequence length reduces activation directly. Activation recomputation lowers the factor from 14 to 2.2."),
        ),
        (
            t(lang, "临时拼参", "Temporary gather"),
            memory["transient_gb"],
            t(lang, "这是 ZeRO-3 或 FSDP 计算当前层时短暂拼回的参数。", "This is the parameter slice ZeRO-3 or FSDP gathers while the current layer runs."),
        ),
    ]
    name, value, advice = max(parts, key=lambda item: item[1])
    notes.append({
        "id": "memory-lead",
        "level": "info",
        "title": t(lang, f"显存里最大的是{name}", f"Largest memory term: {name}"),
        "body": t(lang, f"{name}约 {value:.2f} GB。{advice}", f"{name} is about {value:.2f} GB. {advice}"),
    })
    if not fits:
        notes.append({
            "id": "oom",
            "level": "warn",
            "title": t(lang, "这张卡放不下", "This GPU cannot hold it"),
            "body": t(
                lang,
                (
                    f"峰值 {memory['total_gb']:.2f} GB，容量 {memory['gpu_gb']:.0f} GB。"
                    "要让它跑起来，可以把参数和优化器状态切到更多卡上，减小微批次，或打开激活重计算。"
                ),
                (
                    f"Peak {memory['total_gb']:.2f} GB, capacity {memory['gpu_gb']:.0f} GB. "
                    "To make it fit, shard parameters and optimizer state across more GPUs, shrink the micro-batch, or turn on activation recomputation."
                ),
            ),
        })
    elif memory["total_gb"] >= 0.92 * memory["gpu_gb"]:
        notes.append({
            "id": "tight",
            "level": "warn",
            "title": t(lang, "显存很满", "Memory is tight"),
            "body": t(
                lang,
                "账单已经占到容量的 92% 以上。真实训练里缓存分配器的碎片可能让同样的数字仍然分配失败。",
                "The bill already uses at least 92% of capacity. In a real run, allocator fragmentation can still fail at the same number.",
            ),
        })
    if timing["comm_ms"] > 0 and timing["exposed_comm_ms"] > timing["pure_compute_ms"]:
        notes.append({
            "id": "comm-bound",
            "level": "warn",
            "title": t(lang, "时间主要耗在通信", "Time is mostly communication"),
            "body": t(
                lang,
                (
                    f"通信总量 {timing['comm_ms']:.0f} ms，暴露出来的有 {timing['exposed_comm_ms']:.0f} ms，"
                    f"纯计算 {timing['pure_compute_ms']:.0f} ms。换更宽的节点间网络，或减少跨节点的梯度体积，步时才会明显下降。"
                ),
                (
                    f"Communication totals {timing['comm_ms']:.0f} ms, of which {timing['exposed_comm_ms']:.0f} ms is exposed, "
                    f"against {timing['pure_compute_ms']:.0f} ms of pure compute. The step drops only with a wider inter-node link or a smaller gradient crossing the nodes."
                ),
            ),
        })
    elif timing["comm_ms"] > 0 and timing["exposed_comm_ms"] == 0:
        notes.append({
            "id": "comm-hidden",
            "level": "info",
            "title": t(lang, "通信被计算盖住了", "Compute hides the communication"),
            "body": t(
                lang,
                (
                    f"通信总量 {timing['comm_ms']:.0f} ms，按重叠比例估算后，暴露通信是 0。"
                    "第一步日志仍会写出同步阶段，方便对照真实训练里的 AllReduce 或 AllGather。"
                ),
                (
                    f"Communication totals {timing['comm_ms']:.0f} ms. After the overlap factor, exposed communication is 0. "
                    "The first step still logs a sync phase, so it can be compared with AllReduce or AllGather in a real run."
                ),
            ),
        })
    if tp > 1 and tp_cross:
        notes.append({
            "id": "tp-cross",
            "level": "warn",
            "title": t(lang, "张量并行跨出了节点", "Tensor parallelism leaves the node"),
            "body": t(
                lang,
                "每一层的集合通信都要离开本机。张量并行更适合放在节点内的 NVLink 上，跨节点的切分更常留给流水线或数据并行。",
                "Every layer's collective leaves the machine. Tensor parallelism fits better on intra-node NVLink. Cross-node splits are more often left to pipeline or data parallelism.",
            ),
        })
    elif tp > 1 and not tp_cross:
        notes.append({
            "id": "tp-local",
            "level": "info",
            "title": t(lang, "张量并行留在节点内", "Tensor parallelism stays in the node"),
            "body": t(
                lang,
                f"张量并行度 {tp} 没有跨出一台机器。层内通信走 {gpu['intra_name']}。",
                f"Tensor parallel degree {tp} stays on one machine. Intra-layer communication uses {gpu['intra_name']}.",
            ),
        })
    if pp > 1:
        hot = timing["bubble_frac"] >= 0.3 or cfg.grad_accum < pp
        notes.append({
            "id": "bubble",
            "level": "warn" if hot else "info",
            "title": t(lang, "流水线气泡", "Pipeline bubble"),
            "body": t(
                lang,
                (
                    f"气泡比例 {timing['bubble_frac'] * 100:.0f}%。"
                    f"微批次数按梯度累积 {cfg.grad_accum} 计算，阶段数是 {pp}。"
                    "微批次数多于阶段数时，气泡占比会下降。"
                ),
                (
                    f"Bubble fraction {timing['bubble_frac'] * 100:.0f}%. "
                    f"Microbatch count follows gradient accumulation {cfg.grad_accum}, and there are {pp} stages. "
                    "The bubble shrinks when there are more microbatches than stages."
                ),
            ),
        })
    if cfg.gpus_per_node > 1 and not gpu["has_nvlink"]:
        notes.append({
            "id": "pcie",
            "level": "warn",
            "title": t(lang, "多卡走 PCIe", "Several GPUs share PCIe"),
            "body": t(
                lang,
                f"{gpu['name']} 之间没有 NVLink。节点内集合通信按 PCIe 有效带宽 {gpu['intra_gbps']} GB/s 估算。",
                f"{gpu['name']} has no NVLink between cards. Intra-node collectives use a PCIe effective bandwidth of {gpu['intra_gbps']} GB/s.",
            ),
        })
    if cfg.nodes > 1:
        notes.append({
            "id": "multinode",
            "level": "info",
            "title": t(lang, "多机怎么配", "How the machines are set up"),
            "body": t(
                lang,
                (
                    f"进程组的 master 是 {cluster['nodes'][0]['host']}:29500。"
                    "各节点使用同一套代码和依赖。估算假定每台机器读取自己的数据分片，节点间传递的是梯度或激活。"
                ),
                (
                    f"The process-group master is {cluster['nodes'][0]['host']}:29500. "
                    "Every node uses the same code and dependencies. The estimate assumes each machine reads its own data shard, and only gradients or activations cross the network."
                ),
            ),
        })
    else:
        link_name = _pick(link, lang)
        notes.append({
            "id": "single-node-net",
            "level": "info",
            "title": t(lang, "节点间网络这次闲着", "The inter-node network is idle"),
            "body": t(
                lang,
                f"只有一台机器。左栏里的 {link['name']} 不进入本次估算，卡与卡之间只走 {gpu['intra_name']}。",
                f"There is only one machine. {link_name} in the left column is not part of this estimate. GPUs talk only over {gpu['intra_name']}.",
            ),
        })
    if cfg.activation_checkpointing:
        notes.append({
            "id": "checkpoint-act",
            "level": "info",
            "title": t(lang, "激活重计算", "Activation recomputation"),
            "body": t(
                lang,
                "激活系数按 2.2 估算。本页的计算时间仍按每 token 6 倍参数量，没有另加重新前向的时间。",
                "The activation factor is 2.2. Compute time on this page is still 6 times the parameter count per token, with no extra forward pass added.",
            ),
        })
    if use_offload:
        notes.append({
            "id": "offload",
            "level": "warn",
            "title": t(lang, "优化器在 CPU", "Optimizer on the CPU"),
            "body": t(
                lang,
                f"GPU 上不再计优化器状态，约 {memory['optim_offload_gb']:.2f} GB 经 PCIe 往返，有效带宽按 {PCIE_OFFLOAD_GBPS:.0f} GB/s。步时里的优化器一项会变长。",
                f"Optimizer state is no longer counted on the GPU. About {memory['optim_offload_gb']:.2f} GB crosses PCIe at {PCIE_OFFLOAD_GBPS:.0f} GB/s. The optimizer slice of the step gets longer.",
            ),
        })
    if cfg.grad_accum > 1 and dp > 1 and (framework == "fsdp" or zero_stage == 3):
        notes.append({
            "id": "zero3-accum",
            "level": "info",
            "title": t(lang, "梯度累积会放大通信次数", "Gradient accumulation multiplies communication"),
            "body": t(
                lang,
                "ZeRO-3 和 FSDP 每个微批次都要拼参数。梯度累积增大全局批次的同时，通信次数也按微批次数增加。",
                "ZeRO-3 and FSDP gather parameters on every microbatch. Accumulation grows the global batch, and it also grows the number of collectives.",
            ),
        })
    if not timing["single_gpu_fits"]:
        notes.append({
            "id": "single-fit",
            "level": "info",
            "title": t(lang, "单卡放不下完整模型", "One GPU cannot hold the full model"),
            "body": t(
                lang,
                "加速比的分子仍是单卡完成同样多 token 的计算时间。这个比值用来比较算力和通信，不表示一张卡真能把训练跑完。",
                "The numerator of the speedup is still the compute time for the same tokens on one GPU. The ratio compares compute and communication. It does not mean one GPU can finish the run.",
            ),
        })
    return notes


def _first_step_line(plan: dict, phase: str, cfg: TrainConfig) -> str:
    lang = norm_lang(plan.get("lang"))
    tag = sim(lang)
    if phase == "data":
        return t(
            lang,
            f"{tag}[step 1] 取数 | 每卡微批次 {cfg.micro_batch} | 序列长度 {cfg.seq_len}",
            f"{tag}[step 1] data | micro-batch per GPU {cfg.micro_batch} | sequence length {cfg.seq_len}",
        )
    if phase == "forward":
        return t(
            lang,
            f"{tag}[step 1] 前向 | 本卡层数 {plan['layers_local']}",
            f"{tag}[step 1] forward | layers on this GPU {plan['layers_local']}",
        )
    if phase == "backward":
        return t(
            lang,
            f"{tag}[step 1] 反向 | 梯度先在本地累积 {cfg.grad_accum} 个微批次，再进入同步",
            f"{tag}[step 1] backward | gradients accumulate locally for {cfg.grad_accum} microbatches, then sync",
        )
    if phase == "comm":
        return t(
            lang,
            f"{tag}[step 1] 同步 | {plan['timing']['comm_bus']}",
            f"{tag}[step 1] sync | {plan['timing']['comm_bus']}",
        )
    warm = warmup_steps(cfg.steps)
    current = lr_at(1, cfg.steps, cfg.lr)
    if warm <= 1:
        lr_text = t(lang, f"学习率 {current:.3e}", f"learning rate {current:.3e}")
    else:
        lr_text = t(
            lang,
            f"学习率升温第 1/{warm} 步，本步 {current:.3e}，之后使用 {cfg.lr:.3e}",
            f"learning-rate warmup step 1/{warm}, this step {current:.3e}, then {cfg.lr:.3e}",
        )
    return t(
        lang,
        f"{tag}[step 1] 优化器 | AdamW | {lr_text}",
        f"{tag}[step 1] optimizer | AdamW | {lr_text}",
    )


def _step_summary(metrics: dict, lang: str = "zh") -> str:
    lang = norm_lang(lang)
    tag = sim(lang)
    return t(
        lang,
        (
            f"{tag}[step {metrics['step']}/{metrics['steps']}] loss {metrics['loss']:.3f} | "
            f"lr {metrics['lr']:.3e} | {metrics['tokens_per_s']:,.0f} tok/s | "
            f"step {metrics['step_ms']:.0f} ms（计算 {metrics['compute_ms']:.0f} / "
            f"暴露通信 {metrics['exposed_comm_ms']:.0f} / 取数 {metrics['data_ms']:.0f} / "
            f"优化器 {metrics['optim_ms']:.0f}） | 显存 {metrics['mem_gb']:.2f} GB"
        ),
        (
            f"{tag}[step {metrics['step']}/{metrics['steps']}] loss {metrics['loss']:.3f} | "
            f"lr {metrics['lr']:.3e} | {metrics['tokens_per_s']:,.0f} tok/s | "
            f"step {metrics['step_ms']:.0f} ms (compute {metrics['compute_ms']:.0f} / "
            f"exposed comm {metrics['exposed_comm_ms']:.0f} / data {metrics['data_ms']:.0f} / "
            f"optimizer {metrics['optim_ms']:.0f}) | memory {metrics['mem_gb']:.2f} GB"
        ),
    )


async def _pause(seconds: float, is_disconnected: Optional[Disconnect]) -> bool:
    await asyncio.sleep(seconds)
    if is_disconnected is None:
        return True
    return not await is_disconnected()
