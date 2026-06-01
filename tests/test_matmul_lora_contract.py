# Copyright 2026-present the heiervang-technologies team. All rights reserved.
#
# Source-level contract test for the HT fork's `matmul_lora` kernel.
#
# Why structural and not behavioral:
# `unsloth/kernels/utils.py` imports `triton` and CUDA-dependent symbols at
# module top, so we can't `import` it on a CPU-only CI runner. Behavioral
# tests for this kernel live in upstream Unsloth's GPU CI. What the HT fork
# specifically needs to defend across rebases is the *shape* of this
# function — a fused-LoRA-into-quantized-matmul that upstream does not have.
# If a rebase silently rewrites the dispatch (e.g. drops the Float8 branch
# or the fast_dequantize path), this test fails loudly so we catch it in PR
# review instead of in production.

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "unsloth" / "kernels" / "utils.py"


@pytest.fixture(scope="module")
def utils_source() -> str:
    assert UTILS_PATH.is_file(), f"missing {UTILS_PATH}"
    return UTILS_PATH.read_text()


@pytest.fixture(scope="module")
def matmul_lora_def(utils_source: str) -> ast.FunctionDef:
    tree = ast.parse(utils_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "matmul_lora":
            return node
    pytest.fail("matmul_lora not defined at module level in unsloth/kernels/utils.py")


def test_signature_preserved(matmul_lora_def: ast.FunctionDef) -> None:
    # Positional params in order: X, W, W_quant, A, B, s — with `out` keyword.
    args = matmul_lora_def.args
    pos_names = [a.arg for a in args.args]
    assert pos_names == ["X", "W", "W_quant", "A", "B", "s", "out"], (
        f"matmul_lora signature drifted: got {pos_names}"
    )
    # `out` defaults to None
    assert args.defaults, "matmul_lora `out` should default to None"
    out_default = args.defaults[-1]
    assert isinstance(out_default, ast.Constant) and out_default.value is None


def test_float8_dispatch_branches_present(utils_source: str) -> None:
    # The three branches we depend on inside matmul_lora's body. If upstream
    # rewrites the dispatch, we want a red light, not a silent regression.
    src = _slice_function_source(utils_source, "matmul_lora")
    assert "Float8Tensor" in src, "Float8Tensor dispatch branch missing"
    assert "float8_e4m3fn" in src, "fp8_e4m3fn dispatch branch missing"
    assert "fast_dequantize" in src, "fast_dequantize fallback branch missing"
    assert "fp8_linear" in src, "fp8_linear call missing — fp8 path broken"


def test_lora_delta_applied(utils_source: str) -> None:
    # LoRA application: out += (X @ A) @ (s * B). We check the .t() + addmm_
    # idiom that the fused path uses; if upstream rewrites this to a different
    # primitive, the rebase reviewer needs to confirm semantics by hand.
    src = _slice_function_source(utils_source, "matmul_lora")
    assert "A.t()" in src and "B.t()" in src, "LoRA transpose missing"
    assert "addmm_" in src, "in-place LoRA add missing"
    assert "alpha = s" in src or "alpha=s" in src, "LoRA scaling alpha missing"


def test_3d_reshape_round_trip(utils_source: str) -> None:
    # The 3D input case is what gets called from the model forward. The
    # round-trip (view → call → view) is the contract Studio depends on.
    src = _slice_function_source(utils_source, "matmul_lora")
    assert "X.dim() == 3" in src, "3D-input branch missing"
    assert "X.view(-1, X.shape[-1])" in src, "3D flatten missing"
    assert "view(batch, seq_len, -1)" in src, "3D restore missing"


def _slice_function_source(source: str, fn_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{fn_name} not found")
