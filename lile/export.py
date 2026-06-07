"""Safetensors streaming exporter for lile states."""
from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

from lile.state import ModelState

log = logging.getLogger(__name__)

def stream_safetensors(
    tensor_info: Iterable[tuple[str, list[int], str, Callable[[], torch.Tensor]]],
    file_path: Path
) -> None:
    """Stream tensors layer-by-layer to a safetensors file to avoid OOM."""
    header = {"__metadata__": {"format": "pt"}}
    current_offset = 0
    
    computes = []
    for name, shape, dtype_str, compute_fn in tensor_info:
        numel = 1
        for s in shape:
            numel *= s
        dtype_size = {"BF16": 2, "F16": 2, "F32": 4}[dtype_str]
        
        # Safetensors usually pads to 8 bytes per tensor offset
        padding = (8 - (current_offset % 8)) % 8
        current_offset += padding
        
        byte_size = numel * dtype_size
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [current_offset, current_offset + byte_size]
        }
        computes.append((name, compute_fn, padding))
        current_offset += byte_size
        
    header_bytes = json.dumps(header).encode("utf-8")
    header_padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * header_padding
    
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name, compute_fn, padding in computes:
            if padding > 0:
                f.write(b"\x00" * padding)
            tensor = compute_fn()
            # Must be contiguous and on CPU
            t_bytes = tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            f.write(t_bytes)
            del tensor

def export_model(
    state: ModelState,
    out_dir: Path,
    dtype: str = "bf16",
    merge_mode: str = "fold_all"
) -> None:
    from unsloth.kernels import fast_dequantize, get_lora_parameters_bias
    from peft.tuners.lora import Linear4bit as Peft_Linear4bit
    from peft.tuners.lora import Linear as Peft_Linear
    
    # Try importing bnb
    try:
        from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
    except ImportError:
        class Bnb_Linear4bit: pass
        
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.safetensors"
    
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    st_dtype = {"bf16": "BF16", "fp16": "F16", "fp32": "F32"}[dtype]
    
    def _get_tensor_info():
        # Yields (name, shape, st_dtype, compute_fn)
        
        def _find_transformer(m):
            if hasattr(m, "embed_tokens") and hasattr(m, "layers"):
                return m
            for child in m.children():
                res = _find_transformer(child)
                if res is not None:
                    return res
            return None
            
        def _find_lm_head(m):
            if hasattr(m, "lm_head"):
                return m.lm_head
            for child in m.children():
                res = _find_lm_head(child)
                if res is not None:
                    return res
            return None

        internal_model = _find_transformer(state.model)
        if internal_model is None:
            raise ValueError("Could not find transformer base model with embed_tokens and layers")
        
        # embed_tokens
        def _embed():
            return internal_model.embed_tokens.weight.data.to(target_dtype)
        yield ("model.embed_tokens.weight", list(internal_model.embed_tokens.weight.shape), st_dtype, _embed)
        
        # layers
        LLAMA_WEIGHTS = (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
        )
        LLAMA_LAYERNORMS = (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "self_attn.q_norm", "self_attn.k_norm"
        )
        
        for j, layer in enumerate(internal_model.layers):
            for item in LLAMA_WEIGHTS:
                try:
                    # e.g. layer.self_attn.q_proj
                    parts = item.split(".")
                    proj = layer
                    for p in parts:
                        proj = getattr(proj, p)
                except AttributeError:
                    continue
                    
                name = f"model.layers.{j}.{item}.weight"
                bias_name = f"model.layers.{j}.{item}.bias"
                
                def make_compute_w(p=proj, n=name):
                    def compute_w():
                        bias_out = getattr(p, "bias", None)
                        if isinstance(p, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
                            W, quant_state, A, B, s, bias_out = get_lora_parameters_bias(p)
                            if quant_state is not None:
                                W = fast_dequantize(W, quant_state)
                            W = W.to(torch.float32).t()
                            
                            if merge_mode == "fold_all" and A is not None:
                                W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha=s)
                            
                            W = W.t().to(target_dtype)
                        else:
                            W = p.weight.to(target_dtype)
                            
                        # Apply merged_deltas (residual)
                        res = state.merged_deltas.get(n)
                        if res is not None:
                            W = W.to(torch.float32) + res.to(dtype=torch.float32, device=W.device)
                            W = W.to(target_dtype)
                            
                        return W.cpu()
                    return compute_w
                
                # We need shape.
                if isinstance(proj, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
                    out_features = proj.out_features
                    in_features = proj.in_features
                else:
                    out_features, in_features = proj.weight.shape
                
                yield (name, [out_features, in_features], st_dtype, make_compute_w())
                
                # Bias?
                if getattr(proj, "bias", None) is not None:
                    # In Unsloth they might return bias from get_lora_parameters_bias
                    b_shape = list(proj.bias.shape)
                    def make_compute_b(p=proj):
                        def compute_b():
                            # just return bias
                            return p.bias.to(target_dtype).cpu()
                        return compute_b
                    yield (bias_name, b_shape, st_dtype, make_compute_b())
                    
            for item in LLAMA_LAYERNORMS:
                try:
                    parts = item.split(".")
                    ln = layer
                    for p in parts:
                        ln = getattr(ln, p)
                    w = ln.weight
                except AttributeError:
                    continue
                name = f"model.layers.{j}.{item}.weight"
                def make_compute_ln(p_w=w):
                    return lambda: p_w.data.to(target_dtype).cpu()
                yield (name, list(w.shape), st_dtype, make_compute_ln())
                
        # norm
        def _norm():
            return internal_model.norm.weight.data.to(target_dtype).cpu()
        yield ("model.norm.weight", list(internal_model.norm.weight.shape), st_dtype, _norm)
        
        # lm_head (if not tied)
        lm_head = _find_lm_head(state.model)
        if lm_head is not None:
            # We must check if lm_head is just a Linear or tied
            if getattr(lm_head, "weight", None) is not None and internal_model.embed_tokens.weight.data_ptr() != lm_head.weight.data_ptr():
                def _lm_head():
                    return lm_head.weight.data.to(target_dtype).cpu()
                yield ("lm_head.weight", list(lm_head.weight.shape), st_dtype, _lm_head)
    
    log.info("Streaming model state to %s (dtype=%s, merge_mode=%s)", model_path, dtype, merge_mode)
    stream_safetensors(_get_tensor_info(), model_path)
    
    # Save active adapter side-car if fold_residual_only
    if merge_mode == "fold_residual_only":
        log.info("Saving active adapter sidecar")
        from safetensors.torch import save_file
        active = state.extract_active_adapter()
        # Ensure it's the requested dtype
        active = {k: v.to(target_dtype) for k, v in active.items()}
        save_file(active, out_dir / "active_adapter.safetensors")
        
    # Copy tokenizer and config
    log.info("Copying tokenizer and config")
    state.tokenizer.save_pretrained(out_dir)
    
    # Strip quantization_config so it loads as a standard dense model
    export_config = state.model.config
    export_config.save_pretrained(out_dir)
    config_path = out_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            c = json.load(f)
        if "quantization_config" in c:
            del c["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(c, f, indent=2)
    log.info("Export complete.")

def main():
    parser = argparse.ArgumentParser(description="Export a lile snapshot to safetensors")
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot name to export")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--merge-mode", type=str, default="fold_all", choices=["fold_all", "fold_residual_only"])
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    from lile.snapshot import SnapshotManager
    from lile.config import ServeConfig
    cfg = ServeConfig()
    mgr = SnapshotManager(cfg.snapshots_dir, max_count=cfg.snapshot_max_count, protect=cfg.snapshot_protect)
    
    manifest = mgr.get_manifest(args.snapshot)
    if manifest is None:
        raise ValueError(f"Snapshot {args.snapshot} not found")
        
    log.info("Loading snapshot %s (base: %s)", args.snapshot, manifest.base_model_name)
    state = ModelState.load(
        model_name=manifest.base_model_name,
        lora_rank=manifest.lora_rank,
        lora_alpha=manifest.lora_alpha
    )
    mgr.load(args.snapshot, state)
    
    export_model(state, args.out, dtype=args.dtype, merge_mode=args.merge_mode)

if __name__ == "__main__":
    main()
