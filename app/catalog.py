"""硬件、模型和预设。数值取公开规格的量级，供教学估算，不是厂商基准。"""

from app.i18n import norm_lang


def _pick(item: dict, lang: str, field: str = "name") -> str:
    if lang == "en" and item.get(field + "_en"):
        return item[field + "_en"]
    return item[field]

GPUS = {
    "rtx4090": {
        "name": "RTX 4090 24GB",
        "mem_gb": 24,
        "tflops_bf16": 165,
        "tflops_fp32": 82,
        "membw_gbps": 1000,
        "intra_name": "PCIe",
        "intra_gbps": 16,
        "intra_lat_us": 6,
        "has_nvlink": False,
    },
    "a100_80": {
        "name": "A100 80GB",
        "mem_gb": 80,
        "tflops_bf16": 312,
        "tflops_fp32": 19.5,
        "membw_gbps": 2039,
        "intra_name": "NVLink",
        "intra_gbps": 300,
        "intra_lat_us": 1.5,
        "has_nvlink": True,
    },
    "h100_80": {
        "name": "H100 80GB",
        "mem_gb": 80,
        "tflops_bf16": 989,
        "tflops_fp32": 67,
        "membw_gbps": 3350,
        "intra_name": "NVLink",
        "intra_gbps": 450,
        "intra_lat_us": 1.2,
        "has_nvlink": True,
    },
    "h20_96": {
        "name": "H20 96GB",
        "mem_gb": 96,
        "tflops_bf16": 148,
        "tflops_fp32": 44,
        "membw_gbps": 4000,
        "intra_name": "NVLink",
        "intra_gbps": 400,
        "intra_lat_us": 1.2,
        "has_nvlink": True,
    },
}

INTERCONNECTS = {
    "eth_10": {"name": "10GbE 以太网", "name_en": "10GbE Ethernet", "gbps": 1.0, "lat_us": 40},
    "eth_100": {"name": "100GbE 以太网", "name_en": "100GbE Ethernet", "gbps": 10.0, "lat_us": 15},
    "ib_hdr": {"name": "InfiniBand HDR", "gbps": 20.0, "lat_us": 2.0},
    "ib_ndr": {"name": "InfiniBand NDR", "gbps": 40.0, "lat_us": 1.5},
}

# 参数量是教学用的约数，名字不绑定某一家的精确结构。
MODELS = {
    "gpt2": {
        "name": "GPT-2 124M",
        "params": 124e6,
        "layers": 12,
        "hidden": 768,
        "heads": 12,
    },
    "d1_5b": {
        "name": "1.5B 级解码器",
        "name_en": "1.5B-class decoder",
        "params": 1.5e9,
        "layers": 28,
        "hidden": 1536,
        "heads": 12,
    },
    "d7b": {
        "name": "7B 级解码器",
        "name_en": "7B-class decoder",
        "params": 6.7e9,
        "layers": 32,
        "hidden": 4096,
        "heads": 32,
    },
    "d13b": {
        "name": "13B 级解码器",
        "name_en": "13B-class decoder",
        "params": 13e9,
        "layers": 40,
        "hidden": 5120,
        "heads": 40,
    },
    "d70b": {
        "name": "70B 级解码器",
        "name_en": "70B-class decoder",
        "params": 70e9,
        "layers": 80,
        "hidden": 8192,
        "heads": 64,
    },
}

DATASETS = {
    "none": {
        "name": "不外推整次作业",
        "name_en": "Do not extrapolate the full job",
        "tokens": None,
    },
    "tok_10b": {
        "name": "100 亿 token",
        "name_en": "10 billion tokens",
        "tokens": 10_000_000_000,
    },
    "tok_100b": {
        "name": "1000 亿 token",
        "name_en": "100 billion tokens",
        "tokens": 100_000_000_000,
    },
    "tok_1t": {
        "name": "1 万亿 token",
        "name_en": "1 trillion tokens",
        "tokens": 1_000_000_000_000,
    },
}

FRAMEWORKS = {
    "ddp": {
        "name": "PyTorch DDP",
        "blurb": "每张卡一份完整模型，反向时 AllReduce 同步梯度。",
        "blurb_en": "Each GPU holds a full model and AllReduces gradients on the backward pass.",
    },
    "fsdp": {
        "name": "PyTorch FSDP",
        "blurb": "参数、梯度和优化器状态按数据并行切分，计算前临时拼回一层。",
        "blurb_en": "Parameters, gradients, and optimizer state are sharded across data-parallel ranks. One layer is gathered just before it is computed.",
    },
    "deepspeed": {
        "name": "DeepSpeed ZeRO",
        "blurb": "ZeRO-1/2/3 逐级切分优化器状态、梯度和参数。",
        "blurb_en": "ZeRO-1/2/3 shard optimizer state, then gradients, then parameters.",
    },
    "megatron": {
        "name": "Megatron-LM",
        "blurb": "张量并行、流水线并行和数据并行叠在一起。",
        "blurb_en": "Tensor, pipeline, and data parallelism are stacked together.",
    },
}

PRESETS = [
    {
        "id": "single_small",
        "name": "单卡入门",
        "name_en": "Single-GPU start",
        "summary": "一张 4090 训练 GPT-2，先把单卡上的一步看完整。",
        "summary_en": "One 4090 trains GPT-2, so a full step on a single GPU is visible.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 1,
            "gpu": "rtx4090",
            "interconnect": "eth_10",
            "framework": "ddp",
            "zero_stage": 3,
            "tp": 1,
            "pp": 1,
            "model": "gpt2",
            "micro_batch": 8,
            "grad_accum": 1,
            "seq_len": 1024,
            "precision": "bf16",
            "activation_checkpointing": False,
            "cpu_offload": False,
            "steps": 12,
            "lr": 3e-4,
            "speed": 2,
        },
    },
    {
        "id": "single_node_dp",
        "name": "单机八卡数据并行",
        "name_en": "Eight GPUs, data parallel",
        "summary": "8 张 A100 用 DDP 训练 1.5B，观察节点内的梯度同步。",
        "summary_en": "Eight A100s train a 1.5B model with DDP, to watch gradient sync inside one node.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 8,
            "gpu": "a100_80",
            "interconnect": "ib_hdr",
            "framework": "ddp",
            "zero_stage": 2,
            "tp": 1,
            "pp": 1,
            "model": "d1_5b",
            "micro_batch": 2,
            "grad_accum": 2,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": False,
            "cpu_offload": False,
            "steps": 12,
            "lr": 2e-4,
            "speed": 2,
        },
    },
    {
        "id": "single_node_shard",
        "name": "单机放不下就分片",
        "name_en": "Shard when one GPU is not enough",
        "summary": "4 张 A100 用 FSDP 训练 7B。完整模型超出单卡显存。",
        "summary_en": "Four A100s train a 7B model with FSDP. The full model exceeds one GPU.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 4,
            "gpu": "a100_80",
            "interconnect": "ib_hdr",
            "framework": "fsdp",
            "zero_stage": 3,
            "tp": 1,
            "pp": 1,
            "model": "d7b",
            "micro_batch": 1,
            "grad_accum": 4,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": True,
            "cpu_offload": False,
            "steps": 12,
            "lr": 1.5e-4,
            "speed": 2,
        },
    },
    {
        "id": "multi_node_megatron",
        "name": "多机三维并行",
        "name_en": "Multi-node 3D parallel",
        "summary": "4 个节点、每节点 8 张 H100，Megatron 训练 70B。张量并行留在节点内。",
        "summary_en": "Four nodes of eight H100s train a 70B model with Megatron. Tensor parallelism stays inside the node.",
        "config": {
            "nodes": 4,
            "gpus_per_node": 8,
            "gpu": "h100_80",
            "interconnect": "ib_ndr",
            "framework": "megatron",
            "zero_stage": 3,
            "tp": 8,
            "pp": 2,
            "model": "d70b",
            "micro_batch": 1,
            "grad_accum": 8,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": True,
            "cpu_offload": False,
            "steps": 10,
            "lr": 1.5e-4,
            "speed": 2,
        },
    },
    {
        "id": "slow_network",
        "name": "慢网络拖垮步时",
        "name_en": "A slow network stretches the step",
        "summary": "两台机器用 10GbE 做数据并行。计算不慢，梯度同步会很慢。",
        "summary_en": "Two machines do data parallel over 10GbE. Compute is fine; gradient sync is slow.",
        "config": {
            "nodes": 2,
            "gpus_per_node": 2,
            "gpu": "rtx4090",
            "interconnect": "eth_10",
            "framework": "ddp",
            "zero_stage": 2,
            "tp": 1,
            "pp": 1,
            "model": "gpt2",
            "micro_batch": 4,
            "grad_accum": 1,
            "seq_len": 1024,
            "precision": "bf16",
            "activation_checkpointing": False,
            "cpu_offload": False,
            "steps": 8,
            "lr": 3e-4,
            "speed": 2,
        },
    },
    {
        "id": "oom_case",
        "name": "单卡撑爆 7B",
        "name_en": "One GPU cannot hold 7B",
        "summary": "一张 24GB 卡装 7B 的 Adam 状态。用来看显存账单，而不是把训练跑完。",
        "summary_en": "A 24GB GPU cannot hold the Adam state of a 7B model. The point is the memory bill, not finishing the run.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 1,
            "gpu": "rtx4090",
            "interconnect": "eth_10",
            "framework": "ddp",
            "zero_stage": 3,
            "tp": 1,
            "pp": 1,
            "model": "d7b",
            "micro_batch": 1,
            "grad_accum": 1,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": True,
            "cpu_offload": False,
            "steps": 8,
            "lr": 2e-4,
            "speed": 4,
        },
    },
    {
        "id": "token_budget",
        "name": "千亿 token 要跑多久",
        "name_en": "How long 100 billion tokens take",
        "summary": "4 张 A100 用 FSDP 训练 7B，语料按 1000 亿 token 外推整次作业。播放仍只走十几步。",
        "summary_en": "Four A100s train a 7B model with FSDP. The corpus is 100 billion tokens, so the plan extrapolates the full job. Playback still covers a dozen steps.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 4,
            "gpu": "a100_80",
            "interconnect": "ib_hdr",
            "framework": "fsdp",
            "zero_stage": 3,
            "tp": 1,
            "pp": 1,
            "model": "d7b",
            "micro_batch": 1,
            "grad_accum": 4,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": True,
            "cpu_offload": False,
            "steps": 12,
            "lr": 1.5e-4,
            "speed": 2,
            "dataset": "tok_100b",
        },
    },
    {
        "id": "resume_job",
        "name": "从检查点接着训",
        "name_en": "Resume from a checkpoint",
        "summary": "同一份 7B 作业从第 1000 步恢复，并按间隔把检查点的磁盘时间算出来。",
        "summary_en": "The same 7B job resumes at step 1000, and checkpoint disk time is estimated on the save interval.",
        "config": {
            "nodes": 1,
            "gpus_per_node": 4,
            "gpu": "a100_80",
            "interconnect": "ib_hdr",
            "framework": "fsdp",
            "zero_stage": 3,
            "tp": 1,
            "pp": 1,
            "model": "d7b",
            "micro_batch": 1,
            "grad_accum": 4,
            "seq_len": 2048,
            "precision": "bf16",
            "activation_checkpointing": True,
            "cpu_offload": False,
            "steps": 12,
            "lr": 1.5e-4,
            "speed": 2,
            "dataset": "tok_100b",
            "resume_step": 1000,
            "ckpt_every_steps": 4,
        },
    },
    {
        "id": "hang_rank",
        "name": "集合通信超时",
        "name_en": "Collective timeout",
        "summary": "两台机器用 10GbE 做数据并行。播放停在一次集合通信超时，日志写出 rank 和步号。",
        "summary_en": "Two machines do data parallel over 10GbE. Playback stops on a collective timeout and names the rank and the step.",
        "config": {
            "nodes": 2,
            "gpus_per_node": 2,
            "gpu": "rtx4090",
            "interconnect": "eth_10",
            "framework": "ddp",
            "zero_stage": 2,
            "tp": 1,
            "pp": 1,
            "model": "gpt2",
            "micro_batch": 4,
            "grad_accum": 1,
            "seq_len": 1024,
            "precision": "bf16",
            "activation_checkpointing": False,
            "cpu_offload": False,
            "steps": 8,
            "lr": 3e-4,
            "speed": 2,
            "fault": "nccl_timeout",
        },
    },
]


def catalog_payload(lang: str = "zh") -> dict:
    lang = norm_lang(lang)
    precisions = {
        "zh": [
            {"id": "bf16", "name": "BF16 混合精度"},
            {"id": "fp16", "name": "FP16 混合精度"},
            {"id": "fp32", "name": "FP32"},
        ],
        "en": [
            {"id": "bf16", "name": "BF16 mixed precision"},
            {"id": "fp16", "name": "FP16 mixed precision"},
            {"id": "fp32", "name": "FP32"},
        ],
    }
    return {
        "lang": lang,
        "gpus": [
            {
                "id": key,
                "name": value["name"],
                "mem_gb": value["mem_gb"],
                "intra_name": value["intra_name"],
                "has_nvlink": value["has_nvlink"],
            }
            for key, value in GPUS.items()
        ],
        "interconnects": [
            {"id": key, "name": _pick(value, lang), "gbps": value["gbps"]}
            for key, value in INTERCONNECTS.items()
        ],
        "models": [
            {
                "id": key,
                "name": _pick(value, lang),
                "params": value["params"],
                "layers": value["layers"],
                "hidden": value["hidden"],
                "heads": value["heads"],
            }
            for key, value in MODELS.items()
        ],
        "frameworks": [
            {"id": key, "name": value["name"], "blurb": _pick(value, lang, "blurb")}
            for key, value in FRAMEWORKS.items()
        ],
        "precisions": precisions[lang],
        "datasets": [
            {"id": key, "name": _pick(value, lang), "tokens": value["tokens"]}
            for key, value in DATASETS.items()
        ],
        "presets": [
            {
                "id": item["id"],
                "name": _pick(item, lang),
                "summary": _pick(item, lang, "summary"),
                "config": item["config"],
            }
            for item in PRESETS
        ],
    }
