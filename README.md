[English](README.en.md)

# 大模型训练模拟器

这是一台教学用的训练模拟器。它不占用 GPU，也不启动真实的 PyTorch、DeepSpeed 或 Megatron 进程。在网页上选定集群、模型和框架之后，可以看到显存账单、一步时间的拆分，以及带 `[模拟]` 前缀的日志。

数字用来比较配置、解释时间和显存去了哪里，不是厂商基准，也不是某次真实任务的跑分。

## 页面上能看到什么

- 三种拓扑：一张卡、一个节点上的多张卡、多个节点上的多张卡。
- 四种框架：PyTorch DDP、PyTorch FSDP、DeepSpeed ZeRO-1/2/3（可选把优化器状态放到 CPU）、Megatron-LM 的张量并行、流水线并行和数据并行。
- 六个预设：单卡入门、单机八卡数据并行、单机放不下就分片、多机三维并行、慢网络拖垮步时、单卡撑爆 7B。
- 右上角的 **中文 / EN** 可以切换页面、讲解和模拟日志的语言。刷新之后仍使用这次的选择。

## 运行

需要 Python 3.10，以及 `requirements.txt` 里的 FastAPI 和 Uvicorn。

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

在浏览器打开 <http://127.0.0.1:7860>。

## 测试

```bash
python3 -m unittest tests.test_plan -v
```

## 文档

- [技术方案（中文）](docs/技术方案.md)
- [Technical design (English)](docs/technical-design.md)
