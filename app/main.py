"""网页和接口。页面、预览、播放都由这一个进程提供。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.catalog import catalog_payload
from app.engine import TrainConfig, build_plan, simulate

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="训练模拟台")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class RunConfig(BaseModel):
    nodes: int = Field(ge=1, le=32)
    gpus_per_node: int = Field(ge=1, le=8)
    gpu: str
    interconnect: str
    framework: str
    zero_stage: int = Field(ge=1, le=3)
    tp: int = Field(ge=1, le=64)
    pp: int = Field(ge=1, le=64)
    model: str
    micro_batch: int = Field(ge=1, le=32)
    grad_accum: int = Field(ge=1, le=64)
    seq_len: int = Field(ge=128, le=8192)
    precision: str
    activation_checkpointing: bool
    cpu_offload: bool
    steps: int = Field(ge=1, le=300)
    lr: float = Field(gt=0, le=1)
    speed: float = Field(ge=1, le=8)

    def to_train(self) -> TrainConfig:
        return TrainConfig(**self.model_dump())


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/catalog")
def catalog():
    return catalog_payload()


@app.post("/api/preview")
def preview(cfg: RunConfig):
    return build_plan(cfg.to_train())


@app.post("/api/run")
async def run(cfg: RunConfig, request: Request):
    train = cfg.to_train()

    async def generate():
        try:
            async for event in simulate(train, request.is_disconnected):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
