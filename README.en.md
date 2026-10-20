[中文](README.md)

# LLM training simulator

This is a teaching simulator. It does not use a GPU, and it does not launch real PyTorch, DeepSpeed, or Megatron processes. After you pick a cluster, a model, and a framework on the page, you see a memory bill, a breakdown of one training step, and a log prefixed with `[sim]`.

The numbers compare configurations and show where time and memory go. They are not a vendor benchmark and not the score of a real job.

## What the page shows

- Three topologies: one GPU, several GPUs on one node, and several nodes with several GPUs each.
- Four frameworks: PyTorch DDP, PyTorch FSDP, DeepSpeed ZeRO-1/2/3 (optimizer state can sit on the CPU), and Megatron-LM with tensor, pipeline, and data parallelism.
- Six presets: a single-GPU start, eight GPUs with data parallelism, sharding when one GPU is not enough, multi-node 3D parallelism, a slow network, and a 7B model that does not fit on one GPU.
- **中文 / EN** in the top corner switches the page, the notes, and the simulated log. A refresh keeps that choice.

## Run

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open <http://127.0.0.1:7860>.

## Tests

```bash
uv run python -m unittest tests.test_plan -v
```

## Documents

- [技术方案（中文）](docs/技术方案.md)
- [Technical design (English)](docs/technical-design.md)
