"""锁定技术方案第 11 节的估算结论。"""

from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import replace

from app.catalog import PRESETS, catalog_payload
from app.engine import TrainConfig, build_plan, loss_series, simulate, spike_step, startup_lines


def config_of(preset_id: str) -> TrainConfig:
    raw = next(item["config"] for item in PRESETS if item["id"] == preset_id)
    return TrainConfig(**raw)


def plan_of(preset_id: str) -> dict:
    return build_plan(config_of(preset_id))


class PlanTests(unittest.TestCase):
    def test_same_config_same_plan(self):
        cfg = config_of("single_node_dp")
        self.assertEqual(build_plan(cfg), build_plan(cfg))

    def test_single_gpu_gpt2_fits(self):
        plan = plan_of("single_small")
        self.assertEqual(plan["errors"], [])
        self.assertTrue(plan["memory"]["fits"])
        self.assertTrue(plan["timing"]["single_gpu_fits"])
        self.assertEqual(plan["parallel"]["world"], 1)
        self.assertLess(plan["timing"]["mfu"], 0.5)

    def test_7b_on_one_4090_does_not_fit(self):
        plan = plan_of("oom_case")
        self.assertEqual(plan["errors"], [])
        self.assertFalse(plan["memory"]["fits"])
        self.assertGreater(plan["memory"]["optim_gb"], plan["memory"]["gpu_gb"])

    def test_fsdp_7b_fits_and_single_gpu_does_not(self):
        plan = plan_of("single_node_shard")
        self.assertEqual(plan["errors"], [])
        self.assertTrue(plan["memory"]["fits"])
        self.assertFalse(plan["timing"]["single_gpu_fits"])
        self.assertEqual(plan["parallel"]["dp"], 4)
        self.assertGreater(plan["memory"]["transient_gb"], 0)

    def test_tp_must_divide_heads(self):
        cfg = TrainConfig(
            nodes=1,
            gpus_per_node=5,
            gpu="a100_80",
            interconnect="ib_hdr",
            framework="megatron",
            zero_stage=1,
            tp=5,
            pp=1,
            model="d7b",
            micro_batch=1,
            grad_accum=1,
            seq_len=512,
            precision="bf16",
            activation_checkpointing=False,
            cpu_offload=False,
            steps=2,
            lr=1e-4,
            speed=8,
        )
        plan = build_plan(cfg)
        self.assertTrue(plan["errors"])
        self.assertTrue(any("注意力头数" in err for err in plan["errors"]))

    def test_world_must_divide(self):
        cfg = replace(
            config_of("single_node_dp"),
            framework="megatron",
            tp=3,
            pp=1,
            model="d1_5b",
        )
        plan = build_plan(cfg)
        self.assertTrue(any("不能被张量并行" in err for err in plan["errors"]))

    def test_slow_network_is_comm_bound(self):
        plan = plan_of("slow_network")
        self.assertEqual(plan["errors"], [])
        self.assertTrue(plan["memory"]["fits"])
        self.assertGreater(plan["timing"]["exposed_comm_ms"], plan["timing"]["compute_ms"])

    def test_megatron_tp_stays_inside_node(self):
        plan = plan_of("multi_node_megatron")
        self.assertEqual(plan["errors"], [])
        self.assertTrue(plan["memory"]["fits"])
        self.assertEqual(plan["parallel"], {"world": 32, "dp": 2, "tp": 8, "pp": 2, "zero_stage": 0})
        groups: dict[tuple, set[int]] = {}
        for node in plan["cluster"]["nodes"]:
            for gpu in node["gpus"]:
                groups.setdefault((gpu["dp"], gpu["pp"]), set()).add(node["id"])
        for nodes in groups.values():
            self.assertEqual(len(nodes), 1)

    def test_global_batch(self):
        for preset in PRESETS:
            cfg = TrainConfig(**preset["config"])
            plan = build_plan(cfg)
            if plan["errors"]:
                continue
            dp = plan["parallel"]["dp"]
            self.assertEqual(
                plan["batch"]["global_samples"],
                cfg.micro_batch * cfg.grad_accum * dp,
            )

    def test_startup_marks_simulation_and_oom(self):
        oom_lines = [text for _level, text in startup_lines(plan_of("oom_case"))]
        self.assertTrue(all(text.startswith("[模拟]") for text in oom_lines))
        self.assertTrue(any("分配失败" in text for text in oom_lines))
        self.assertTrue(any("torchrun" in text for text in oom_lines))

        small_lines = [text for _level, text in startup_lines(plan_of("single_small"))]
        self.assertTrue(any("进入训练循环" in text for text in small_lines))
        self.assertFalse(any("分配失败" in text for text in small_lines))

    def test_loss_series_is_repeatable_and_spikes_only_multinode(self):
        cfg = config_of("slow_network")
        self.assertEqual(loss_series(cfg), loss_series(cfg))
        self.assertEqual(spike_step(cfg), cfg.steps // 2)
        self.assertIsNone(spike_step(config_of("single_small")))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_parallel_does_not_train(self):
        cfg = replace(config_of("multi_node_megatron"), tp=5, speed=8)
        events = []
        async for event in simulate(cfg):
            events.append(event)
        statuses = [event["status"] for event in events if event["kind"] == "status"]
        self.assertIn("error", statuses)
        self.assertFalse(any(event["kind"] == "metrics" for event in events))

    async def test_short_run_finishes(self):
        cfg = replace(config_of("single_small"), steps=1, speed=8)
        events = []
        async for event in simulate(cfg):
            events.append(event)
        self.assertEqual(events[0]["kind"], "status")
        self.assertEqual(events[1]["kind"], "plan")
        self.assertTrue(any(event["kind"] == "log" and "torchrun" in event["text"] for event in events))
        self.assertTrue(any(event["kind"] == "metrics" for event in events))
        self.assertEqual(events[-1], {"kind": "status", "status": "done"})

    async def test_oom_stream_stops_before_steps(self):
        cfg = replace(config_of("oom_case"), speed=8)
        events = []
        async for event in simulate(cfg):
            events.append(event)
        self.assertEqual(events[-1], {"kind": "status", "status": "oom"})
        self.assertFalse(any(event["kind"] == "metrics" for event in events))
        self.assertTrue(any("分配失败" in event.get("text", "") for event in events))


class LanguageTests(unittest.TestCase):
    def test_chinese_copy_stays_the_default(self):
        plan = plan_of("oom_case")
        self.assertEqual(plan["lang"], "zh")
        lines = [text for _level, text in startup_lines(plan)]
        self.assertTrue(all(text.startswith("[模拟]") for text in lines))
        self.assertTrue(any("分配失败" in text for text in lines))
        self.assertEqual(catalog_payload()["presets"][0]["name"], "单卡入门")

    def test_english_copy_is_available(self):
        cfg = config_of("oom_case")
        plan = build_plan(cfg, "en")
        self.assertEqual(plan["lang"], "en")
        self.assertTrue(any("This GPU cannot hold it" == note["title"] for note in plan["notes"]))
        lines = [text for _level, text in startup_lines(plan)]
        self.assertTrue(all(text.startswith("[sim]") for text in lines))
        self.assertTrue(any("Allocation failed" in text for text in lines))
        self.assertFalse(any("分配失败" in text for text in lines))
        names = {item["id"]: item["name"] for item in catalog_payload("en")["presets"]}
        self.assertEqual(names["single_small"], "Single-GPU start")
        bad = build_plan(replace(cfg, framework="megatron", tp=5, pp=1, gpus_per_node=5), "en")
        self.assertTrue(any("attention heads" in err.lower() or "Attention heads" in err for err in bad["errors"]))


class JobTests(unittest.TestCase):
    def test_full_job_is_tokens_divided_by_one_step(self):
        plan = plan_of("token_budget")
        tokens = plan["batch"]["global_tokens"]
        self.assertEqual(plan["job"]["full_steps"], math.ceil(1e11 / tokens))
        step_s = plan["timing"]["step_ms"] / 1000
        world = plan["parallel"]["world"]
        self.assertAlmostEqual(plan["job"]["gpu_hours"], world * plan["job"]["full_steps"] * step_s / 3600)
        self.assertEqual(plan["timing"], plan_of("single_node_shard")["timing"])
        self.assertEqual(plan["memory"], plan_of("single_node_shard")["memory"])

    def test_packing_and_cosine_do_not_change_the_step(self):
        base = config_of("single_small")
        packed = build_plan(replace(base, pack_sequences=True))
        padded = build_plan(replace(base, pack_sequences=False))
        cosine = build_plan(replace(base, lr_schedule="warmup_cosine"))
        self.assertEqual(packed["timing"], padded["timing"])
        self.assertEqual(packed["memory"], padded["memory"])
        self.assertEqual(packed["job"]["effective_token_ratio"], 1.0)
        self.assertEqual(padded["job"]["effective_token_ratio"], 0.7)
        self.assertEqual(cosine["timing"], packed["timing"])
        self.assertEqual(cosine["memory"], packed["memory"])

    def test_checkpoint_time_stays_out_of_the_preview_step(self):
        base = plan_of("single_node_shard")
        saved = plan_of("resume_job")
        self.assertEqual(saved["timing"]["step_ms"], base["timing"]["step_ms"])
        self.assertGreater(saved["job"]["checkpoint"]["ckpt_ms"], 0)
        self.assertEqual(saved["job"]["resume_step"], 1000)
        lines = [text for _level, text in startup_lines(saved)]
        self.assertTrue(any("从第 1000 步恢复" in text for text in lines))
        self.assertTrue(any("进入训练循环" in text for text in lines))

    def test_tensor_parallel_across_nodes_is_advice(self):
        cfg = replace(
            config_of("multi_node_megatron"),
            nodes=2,
            gpus_per_node=4,
            tp=8,
            pp=1,
            model="d7b",
            micro_batch=1,
            grad_accum=1,
            seq_len=512,
        )
        plan = build_plan(cfg)
        self.assertEqual(plan["errors"], [])
        self.assertTrue(any(item["level"] == "warn" and "跨出节点" in item["body"] for item in plan["advice"]))

    def test_old_presets_keep_default_job_fields(self):
        plan = plan_of("single_small")
        self.assertIsNone(plan["job"]["full_steps"])
        self.assertEqual(plan["job"]["lr_schedule"], "warmup_constant")
        self.assertEqual(plan["job"]["fault"], "none")


class JobStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_fp16_overflow_skips_the_update(self):
        cfg = replace(config_of("single_small"), precision="fp16", steps=4, speed=8)
        texts = []
        async for event in simulate(cfg):
            if event["kind"] == "log":
                texts.append(event["text"])
        self.assertTrue(any("梯度溢出" in text for text in texts))
        bf16 = []
        async for event in simulate(replace(cfg, precision="bf16")):
            if event["kind"] == "log":
                bf16.append(event["text"])
        self.assertFalse(any("梯度溢出" in text for text in bf16))

    async def test_collective_timeout_ends_the_run(self):
        cfg = replace(config_of("hang_rank"), speed=8)
        events = []
        async for event in simulate(cfg):
            events.append(event)
        statuses = [event["status"] for event in events if event["kind"] == "status"]
        self.assertEqual(statuses[-1], "timeout")
        self.assertNotIn("done", statuses)
        text = " ".join(event.get("text", "") for event in events)
        self.assertIn("allreduce", text)
        self.assertIn("超时", text)
        self.assertIn("rank", text)


if __name__ == "__main__":
    unittest.main()
