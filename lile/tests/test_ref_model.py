"""Tests for the frozen reference model wiring (W1).

Covers:
* ``ControllerConfig.frozen_ref`` defaults to False (backwards-compat).
* When False, ``Controller._ref_model is Controller.state.model`` (EMA-1 alias).
* When True, ``Controller._ref_model`` is a distinct object loaded via
  ``LiveState.load_frozen_ref()``, with all params frozen.

The unit tests here use a fake ``LiveState`` so they run without CUDA. A CUDA-gated
end-to-end check lives in ``test_smoke.py`` so the wiring is exercised against a
real model on the GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lile.controller import Controller, ControllerConfig
from lile.state import StateConfig


class _FakeLiveState:
    """Stand-in for LiveState that doesn't touch CUDA."""

    def __init__(self, config):
        self.config = config
        self.model = nn.Linear(4, 4)  # one trainable param so TrainEngine accepts it
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 0
        self.merged_deltas = {}
        self.merge_count = 0
        self._frozen_ref_calls = 0

    def load(self):
        return self

    def load_frozen_ref(self):
        self._frozen_ref_calls += 1
        ref = nn.Linear(4, 4)
        for p in ref.parameters():
            p.requires_grad = False
        ref.eval()
        return ref

    def vram_summary(self):
        return {"allocated_gb": 0.0, "peak_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}

    def set_inference_mode(self):
        self.model.eval()

    def set_training_mode(self):
        self.model.train()


def _patched_controller(monkeypatch, *, frozen_ref):
    cfg = ControllerConfig(
        state=StateConfig(model_name="fake/model"),
        work_dir="/tmp/lile-test-ref",
        frozen_ref=frozen_ref,
    )
    # Patch the LiveState the controller imports.
    fake = _FakeLiveState(cfg.state)
    monkeypatch.setattr("lile.controller.LiveState", lambda c: fake)
    # AdapterManager + InferenceEngine + TrainEngine all call into model;
    # mock the heavyweight ones.
    monkeypatch.setattr("lile.controller.AdapterManager", lambda *a, **k: MagicMock())
    monkeypatch.setattr("lile.controller.InferenceEngine", lambda *a, **k: MagicMock())
    return Controller(cfg), fake


def test_config_default_frozen_ref_is_false():
    cfg = ControllerConfig(state=StateConfig(model_name="x"))
    assert cfg.frozen_ref is False


def test_no_frozen_ref_aliases_live_model(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.mkdir", lambda *a, **k: None)
    c, fake = _patched_controller(monkeypatch, frozen_ref=False)
    c.work_dir = tmp_path
    c.start()
    try:
        assert c._ref_model is c.state.model
        assert fake._frozen_ref_calls == 0
    finally:
        c.shutdown()


def test_frozen_ref_loads_distinct_model(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.mkdir", lambda *a, **k: None)
    c, fake = _patched_controller(monkeypatch, frozen_ref=True)
    c.work_dir = tmp_path
    c.start()
    try:
        assert c._ref_model is not c.state.model
        assert fake._frozen_ref_calls == 1
        assert all(not p.requires_grad for p in c._ref_model.parameters())
        assert not c._ref_model.training  # eval mode
    finally:
        c.shutdown()
