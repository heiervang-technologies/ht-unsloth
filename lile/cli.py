"""``lile`` command-line entrypoint.

Subcommands:

* ``lile serve`` — boot the daemon. Loads the model (this is the slow step,
  ~15 s for Qwen3-0.6B 4-bit on a 3090, ~60 s for 7B), wires up the controller,
  and runs uvicorn until killed.
* ``lile bench-ranking`` — re-run the §11 ranking-reliability benchmark.
* ``lile sanity`` — load + 1 generation, print VRAM and timing. The first thing
  to run on a new machine.

We deliberately keep this a flat argparse CLI: hot-reload, daemon mode, etc.
are systemd's job, not ours.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any


_LOG = logging.getLogger("lile.cli")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_controller(args: argparse.Namespace):
    """Construct and start a :class:`Controller`. Imports unsloth here, not at
    module top level, so ``lile --help`` works without CUDA."""
    from lile.controller import Controller, ControllerConfig
    from lile.state import StateConfig

    cfg = ControllerConfig(
        state=StateConfig(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
            full_finetuning=args.full_finetuning,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            inference_backend=getattr(args, "inference_backend", "fast_generate"),
            sidecar_mode=getattr(args, "sidecar_mode", "colocate"),
            sidecar_device=getattr(args, "sidecar_device", "cuda:1"),
            sidecar_gpu_memory_utilization=getattr(
                args, "sidecar_gpu_memory_utilization", 0.4,
            ),
        ),
        work_dir=args.work_dir,
        lr=args.lr,
        frozen_ref=getattr(args, "frozen_ref", False),
        idle_replay=getattr(args, "idle_replay", False),
        idle_replay_threshold_s=getattr(args, "idle_threshold_s", 30.0),
    )
    return Controller(cfg).start()


def cmd_serve(args: argparse.Namespace) -> int:
    """``lile serve`` — start the FastAPI daemon."""
    _setup_logging(args.log_level)
    import uvicorn  # noqa: PLC0415

    _LOG.info("loading model %s …", args.model)
    controller = _build_controller(args)
    _LOG.info("model loaded; vram=%s", controller.state.vram_summary())

    from lile.server import create_app  # noqa: PLC0415

    app = create_app(controller)

    # Graceful shutdown: drain queue + close trajectory log on SIGINT/SIGTERM.
    def _on_signal(signum, frame):  # noqa: ARG001
        _LOG.info("signal %s → shutting down controller", signum)
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=args.access_log,
        # Single worker process: the controller owns one CUDA context.
        workers=1,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        controller.shutdown()
    return 0


def cmd_sanity(args: argparse.Namespace) -> int:
    """``lile sanity`` — quick load + generate sanity check."""
    _setup_logging(args.log_level)
    controller = _build_controller(args)
    try:
        result = controller.chat(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            max_new_tokens=32,
            temperature=0.7,
            do_sample=True,
        )
        print(f"--- generation ({result.completion_tokens} tok in {result.elapsed_s:.2f}s) ---")
        print(result.text.strip())
        print("--- vram ---")
        for k, v in controller.state.vram_summary().items():
            print(f"  {k} = {v:.3f}")
    finally:
        controller.shutdown()
    return 0


def cmd_bench_ranking(args: argparse.Namespace) -> int:
    """``lile bench-ranking`` — re-run the §11 benchmark, printing summary."""
    _setup_logging(args.log_level)
    # Run the benchmark module's main() so the script and the CLI stay in sync.
    mod = importlib.import_module("benchmarks.ccpd_ranking_reliability")
    return int(mod.main(model=args.model, k=args.k, beta=args.beta))


# --- main -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lile", description="LiveLearn daemon.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args used by serve + sanity + bench.
    def _add_model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--model",
            default=os.environ.get(
                "LILE_MODEL",
                "unsloth/qwen3-0.6b-unsloth-bnb-4bit",
            ),
        )
        sp.add_argument("--max-seq-length", type=int, default=2048)
        sp.add_argument("--no-4bit", action="store_true",
                        help="Disable bnb 4-bit loading (use bf16/fp16 base).")
        sp.add_argument("--full-finetuning", action="store_true",
                        help="Disable LoRA; full-FT mode.")
        sp.add_argument("--lora-rank", type=int, default=16)
        sp.add_argument("--lora-alpha", type=int, default=16)
        sp.add_argument("--lr", type=float, default=1e-5)
        sp.add_argument("--work-dir", default="./.lile")
        sp.add_argument(
            "--frozen-ref", dest="frozen_ref", action="store_true",
            help="Load a second base-model copy as π_ref (recommended for 7B+; "
                 "kills the EMA-1 self-reference weakness in KL/KTO/CCPD).",
        )
        sp.add_argument(
            "--idle-replay", dest="idle_replay", action="store_true",
            help="Enable T4.1 idle replay scheduler — replays past feedback "
                 "during quiet periods to keep the GPU warm.",
        )
        sp.add_argument(
            "--idle-threshold-s", dest="idle_threshold_s", type=float, default=30.0,
            help="Seconds the queue must be idle before the scheduler "
                 "considers replaying. Default 30.",
        )
        sp.add_argument(
            "--inference-backend", dest="inference_backend",
            choices=["fast_generate", "vllm_sidecar"], default="fast_generate",
            help="Inference path. fast_generate (default) shares the trainer's "
                 "CUDA context (chat blocks training). vllm_sidecar runs vLLM "
                 "concurrently and syncs adapters at merge boundaries — "
                 "requires `pip install vllm`.",
        )
        sp.add_argument(
            "--sidecar-mode", dest="sidecar_mode",
            choices=["colocate", "separate"], default="colocate",
            help="vLLM sidecar deployment. colocate shares one GPU with the "
                 "trainer (cudaIpc weight sync). separate puts vLLM on a "
                 "second GPU (NCCL weight sync; requires ≥2 GPUs).",
        )
        sp.add_argument(
            "--sidecar-device", dest="sidecar_device", default="cuda:1",
            help="Device for the vLLM sidecar in separate mode. Ignored in "
                 "colocate mode. Default cuda:1.",
        )
        sp.add_argument(
            "--sidecar-gpu-memory-utilization", dest="sidecar_gpu_memory_utilization",
            type=float, default=0.4,
            help="Fraction of the sidecar GPU's free VRAM to use for "
                 "PagedAttention. In colocate mode this is taken from the "
                 "shared GPU; leave headroom for the trainer. Default 0.4.",
        )

    sp_serve = sub.add_parser("serve", help="Start the FastAPI daemon.")
    _add_model_args(sp_serve)
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--access-log", action="store_true")
    sp_serve.set_defaults(func=cmd_serve)

    sp_sanity = sub.add_parser("sanity", help="Load model + generate one sentence.")
    _add_model_args(sp_sanity)
    sp_sanity.set_defaults(func=cmd_sanity)

    sp_bench = sub.add_parser("bench-ranking", help="Re-run the §11 benchmark.")
    sp_bench.add_argument(
        "--model",
        default=os.environ.get("LILE_MODEL", "unsloth/qwen3-0.6b-unsloth-bnb-4bit"),
    )
    sp_bench.add_argument("--k", type=int, default=8)
    sp_bench.add_argument("--beta", type=float, default=0.1)
    sp_bench.set_defaults(func=cmd_bench_ranking)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
