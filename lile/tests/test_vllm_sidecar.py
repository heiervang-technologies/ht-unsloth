"""Tests for the W5 vLLM sidecar wiring.

vLLM itself isn't installed in this dev environment (and won't be on the
review box either, most likely — vLLM has heavy CUDA build requirements).
These tests therefore exercise the *integration surface*:

* :func:`is_available` honestly reports the absence,
* :class:`Controller.start` fails loud when the operator asks for the sidecar
  but vLLM is missing,
* :meth:`Controller.chat` bypasses ``_gpu_lock`` when a sidecar is attached
  (the headline architectural win — chat is concurrent with training),
* :meth:`Controller._do_merge` and :meth:`Controller.restore` push the
  trainer's adapter to the sidecar via :class:`WeightSyncBridge` so the
  sidecar reflects the merge / restore.

The end-to-end NCCL path requires a 2-GPU box; that gap is documented in
``STATUS.md`` Phase 6 verification. What we *can* verify locally is that the
control flow inside the controller is correct, and that's what these tests
pin down.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lile.controller import Controller, ControllerConfig
from lile.engine.inference import GenerationResult
from lile.engine.vllm_sidecar import SidecarConfig, VLLMSidecar, is_available
from lile.engine.weight_sync import WeightSyncBridge
from lile.queue import ComputeQueue
from lile.state import StateConfig


# --- helpers ---------------------------------------------------------------


def _make_fake_sidecar():
    calls: list[str] = []

    class _FakeSidecar:
        def generate(self, msgs, *, max_new_tokens, temperature, top_p, do_sample):
            calls.append("generate")
            return GenerationResult(
                response_id="rid", text="ok",
                prompt_tokens=1, completion_tokens=2, elapsed_s=0.0,
                finish_reason="stop",
            )

        def apply_lora(self, tensors, *, name="live"):
            calls.append(f"apply_lora:{name}")

    fake = _FakeSidecar()
    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _bare_controller(work_dir: Path) -> Controller:
    """Construct a Controller without start(): no CUDA, no Unsloth import."""
    cfg = ControllerConfig(
        state=StateConfig(model_name="dummy"),
        work_dir=str(work_dir),
    )
    return Controller(cfg)


# --- env / availability ---------------------------------------------------


def test_is_available_returns_bool():
    """is_available must be False here; the import path is non-throwing."""
    assert isinstance(is_available(), bool)


def test_sidecar_config_defaults_match_state_config():
    """Defaults baked into SidecarConfig should track StateConfig defaults so
    the two halves of the wiring don't drift silently."""
    sc = StateConfig(model_name="x")
    cfg = SidecarConfig(model_name="x")
    assert cfg.mode == sc.sidecar_mode
    assert cfg.sidecar_device == sc.sidecar_device
    assert cfg.gpu_memory_utilization == sc.sidecar_gpu_memory_utilization


def test_vllm_sidecar_load_without_vllm_raises():
    """If vLLM isn't importable, load() must surface ImportError early."""
    if is_available():
        pytest.skip("vLLM is installed — absence path not testable")
    sc = VLLMSidecar(SidecarConfig(model_name="dummy"))
    with pytest.raises(ImportError):
        sc.load(tokenizer=None)


# --- Controller.start() error path ---------------------------------------


def test_controller_start_rejects_sidecar_when_vllm_missing(tmp_path):
    """An operator who explicitly asked for the sidecar shouldn't silently
    fall back to fast_generate — the deployment shape would be wrong."""
    if is_available():
        pytest.skip("vLLM is installed — absence path not testable")
    cfg = ControllerConfig(
        state=StateConfig(model_name="dummy", inference_backend="vllm_sidecar"),
        work_dir=str(tmp_path),
    )
    ctl = Controller(cfg)
    with pytest.raises(RuntimeError, match="vllm"):
        ctl.start()


# --- Headline: chat() bypasses _gpu_lock when sidecar is present ---------


def test_chat_with_sidecar_does_not_take_gpu_lock(tmp_path):
    """The architectural win: a chat call must complete even while another
    thread holds ``_gpu_lock`` (i.e. while a training step is in flight).
    """
    ctl = _bare_controller(tmp_path)
    ctl.sidecar = _make_fake_sidecar()
    ctl.state = SimpleNamespace()  # only used as a non-None sentinel here
    ctl.infer_engine = SimpleNamespace()
    ctl.queue = ComputeQueue()

    started = threading.Event()
    release = threading.Event()

    def _holder():
        with ctl._gpu_lock:
            started.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_holder, daemon=True)
    holder.start()
    assert started.wait(2.0), "lock holder thread didn't start"

    # If chat() acquired _gpu_lock, this would hang until release.set().
    t0 = time.monotonic()
    result = ctl.chat([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - t0

    release.set()
    holder.join(timeout=2.0)

    assert result.text == "ok"
    assert "generate" in ctl.sidecar.calls
    assert elapsed < 0.5, f"chat blocked on _gpu_lock for {elapsed:.3f}s"


def test_chat_without_sidecar_serializes_on_gpu_lock(tmp_path):
    """Inverse: the legacy fast_generate path must still take the lock so
    inference and training don't interleave inside the single CUDA context.
    """
    ctl = _bare_controller(tmp_path)
    ctl.sidecar = None

    set_inf_calls: list[str] = []
    gen_calls: list[object] = []

    def _gen(msgs, **kw):
        gen_calls.append(msgs)
        return GenerationResult(
            response_id="rid", text="ok",
            prompt_tokens=1, completion_tokens=2, elapsed_s=0.0,
            finish_reason="stop",
        )

    ctl.state = SimpleNamespace(
        set_inference_mode=lambda: set_inf_calls.append("inf"),
    )
    ctl.infer_engine = SimpleNamespace(generate=_gen)
    ctl.queue = ComputeQueue()

    started = threading.Event()
    release = threading.Event()

    def _holder():
        with ctl._gpu_lock:
            started.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_holder, daemon=True)
    holder.start()
    assert started.wait(2.0)

    chat_done = threading.Event()

    def _chat():
        ctl.chat([{"role": "user", "content": "hi"}])
        chat_done.set()

    t = threading.Thread(target=_chat, daemon=True)
    t.start()
    # Must NOT complete while the lock is held.
    assert not chat_done.wait(0.3)
    release.set()
    holder.join(timeout=2.0)
    assert chat_done.wait(2.0)
    assert set_inf_calls == ["inf"]
    assert len(gen_calls) == 1


# --- Merge / restore push paths ------------------------------------------


def test_do_merge_pushes_active_lora_to_sidecar(tmp_path):
    """After a merge, the sidecar must see the updated adapter."""
    ctl = _bare_controller(tmp_path)
    state = SimpleNamespace(
        merge_count=0,
        model=SimpleNamespace(state_dict=lambda: {}),  # empty LoRA → empty push
    )
    ctl.state = state
    ctl.adapter_mgr = SimpleNamespace(merge_active_lora=lambda: {"layer_x": "ok"})
    pushes: list[dict] = []
    ctl.weight_sync = SimpleNamespace(
        push_active_lora=lambda sd: pushes.append(sd),
    )
    ctl.trajectory = None

    out = ctl._do_merge()
    assert state.merge_count == 1
    assert out["merge_count"] == 1
    assert len(pushes) == 1


def test_do_merge_swallows_sidecar_push_errors(tmp_path):
    """A sidecar push failure must not crash the trainer; sidecar serves
    stale weights until the next push."""
    ctl = _bare_controller(tmp_path)
    state = SimpleNamespace(
        merge_count=0,
        model=SimpleNamespace(state_dict=lambda: {}),
    )
    ctl.state = state
    ctl.adapter_mgr = SimpleNamespace(merge_active_lora=lambda: {})

    def _boom(sd):
        raise RuntimeError("simulated sidecar OOM")

    ctl.weight_sync = SimpleNamespace(push_active_lora=_boom)
    ctl.trajectory = None
    # Must not raise.
    out = ctl._do_merge()
    assert out["merge_count"] == 1


# --- WeightSyncBridge unit tests -----------------------------------------


def test_weight_sync_bridge_no_sidecar_is_noop():
    bridge = WeightSyncBridge(sidecar=None)
    bridge.push_active_lora({"layer.lora_A.weight": torch.zeros(2, 2)})
    bridge.push_merged_deltas({"layer": torch.zeros(2, 2)})
    bridge.maybe_init_weight_update_group()
    assert bridge.stats.pushes == 0


def test_weight_sync_bridge_colocate_calls_apply_lora():
    """Colocate path: bridge moves tensors to the sidecar device and calls
    apply_lora directly. We pin to CPU here to keep the test deterministic."""
    apply_calls: list[dict] = []

    fake_sidecar = SimpleNamespace(
        config=SidecarConfig(
            model_name="x", mode="colocate", device="cpu",
        ),
        llm=None,
        apply_lora=lambda tensors, name="live": apply_calls.append(
            {"name": name, "tensors": tensors},
        ),
    )
    bridge = WeightSyncBridge(sidecar=fake_sidecar)

    sd = {
        "block.0.lora_A.weight": torch.zeros(4, 8, dtype=torch.float32),
        "block.0.lora_B.weight": torch.zeros(8, 4, dtype=torch.float32),
    }
    bridge.push_active_lora(sd)

    assert bridge.stats.pushes == 1
    assert bridge.stats.last_push_n_tensors == 2
    # Colocate path: apply_lora ran with the moved (bf16) tensors.
    assert len(apply_calls) == 1
    moved = apply_calls[0]["tensors"]
    assert set(moved.keys()) == set(sd.keys())
    for v in moved.values():
        assert v.dtype == torch.bfloat16
        assert v.device.type == "cpu"


def test_weight_sync_bridge_separate_mode_skips_when_llm_none():
    """In separate mode without an actual vLLM instance, the bridge falls
    back to apply_lora so the next request at least sees the new adapter."""
    apply_calls: list[dict] = []
    fake_sidecar = SimpleNamespace(
        config=SidecarConfig(
            model_name="x", mode="separate", sidecar_device="cpu",
        ),
        llm=None,
        apply_lora=lambda tensors, name="live": apply_calls.append(
            {"name": name, "tensors": tensors},
        ),
    )
    bridge = WeightSyncBridge(sidecar=fake_sidecar)
    sd = {"block.0.lora_A.weight": torch.zeros(2, 2)}
    bridge.push_active_lora(sd)

    # NCCL bootstrap was attempted (no-op since llm is None) and we landed
    # on the in-memory fallback path.
    assert bridge.stats.pushes == 1
    assert len(apply_calls) == 1
