"""safety_monitor — observational Razin-safety sidecar (Task #20).

Cleo's razin-safety-sharpened.md characterization theorem identifies the
non-target tokens that can *grow* under a one-step SFT-family update:

    q_j > p_j   ⟺   p_j < M_p(η)       where
    M_π(η) := -(1/η) · log Σ_k π_k · exp(η · (𝟙[k=t] - π_k))

The unsafe regime is at *small* η — counterintuitive (it inverts the
"smaller LR = safer" heuristic). In the household-AI loop, a correction
"never say Voldemort" can silently lift ``p("Sauron")`` at small η; there
is no aggregate-safety signal to catch that.

``safety_monitor`` is a batch-objective with ``weight=0.0`` semantics. It:

- reads ``target_positions`` / ``target_token_ids`` / ``input_ids`` /
  ``attention_mask`` from the preceding main objective's result dict
  (``TrainEngine.step`` plumbs them through),
- runs one no-grad forward pass on the same batch,
- computes ``M_π(η)`` per (sample, position) and the grower set
  ``{j ≠ t : π_j < M_π(η)}``,
- intersects with the three-tier watchlist
  (daemon-global ∪ batch-level ∪ per-sample), flags alarms.

**Known approximation (AdamW scope).** ``M_p(η)`` is derived from a
plain-SGD logit update. AdamW's (m, v) warp the per-coordinate step so
that Δz_k ≠ η · (𝟙[k=t] - p_k) in general. First-order directions agree;
magnitudes differ. Consequence:

- Alarm fires → definitely unsafe (the SGD-theoretic bound is violated
  and AdamW's first-order agreement ensures at least proportional
  displacement).
- Alarm silent → within the SGD-theoretic safe zone, but NOT a safety
  guarantee under AdamW.

Treat ``M_p(η)`` as a lower-bound heuristic under AdamW, not an exact
bound. Sharper AdamW-specific bounds are follow-up work, not this PR.

Missing ``target_positions`` contract: if the primitive is dispatched
without them (main objective has not opted in), raise ``RuntimeError``
with the pinned message. Primitive orthogonality: main objective owns
position geometry, sidecar owns the Razin-safety computation.
"""
from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F


_MISSING_TARGETS_MSG = (
    "safety_monitor requires target_positions from the preceding main "
    "objective; the objective does not yet expose them. "
    "See safety-monitor-primitive.md."
)


def _m_p_value(pi: torch.Tensor, t: int, eta: float) -> float:
    """Cleo's M_π(η) = -(1/η) · log Σ_k π_k · exp(η · (𝟙[k=t] - π_k)).

    Computed in log-space for numerical stability:
    ``logsumexp(log π + η · (𝟙[·=t] - π))`` — same answer, safe for tiny
    tail mass. ``pi`` is a ``(V,)`` probability tensor on any device/dtype.
    """
    V = pi.size(0)
    indicator = torch.zeros(V, dtype=pi.dtype, device=pi.device)
    indicator[t] = 1.0
    log_pi = pi.clamp_min(1e-45).log()                                 # avoid -inf
    beta = indicator - pi
    log_Z = torch.logsumexp(log_pi + eta * beta, dim=0)
    return float(-(1.0 / eta) * log_Z)


def _resolve_watchlist(
    samples: list[dict[str, Any]],
    batch_watchlist: list[int] | None,
    default_watchlist: list[int] | None,
) -> list[set[int]]:
    """Three-tier watchlist UNION — same shape as kl_anchor's exclude_ids.

    Per-sample ``watchlist`` → session-level correction.
    Batch-level ``watchlist`` → batch-wide user/session policy.
    Daemon-global ``default_watchlist`` → absolute-never floor.
    """
    base: set[int] = set()
    if default_watchlist:
        base.update(int(x) for x in default_watchlist)
    if batch_watchlist:
        base.update(int(x) for x in batch_watchlist)
    out: list[set[int]] = []
    for s in samples:
        w = set(base)
        sw = s.get("watchlist")
        if sw:
            w.update(int(x) for x in sw)
        out.append(w)
    return out


def safety_monitor_loss(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    target_positions: list[list[int]] | None = None,
    target_token_ids: list[list[int]] | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    watchlist: list[int] | None = None,
    default_watchlist: list[int] | None = None,
    alarm_threshold: float = 1.0,
    weight: float = 0.0,
    effective_lr: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Observational Razin-safety sidecar. See module docstring.

    ``weight`` is coerced to 0.0 — the primitive is observational and
    must not contribute to the backward pass. A non-zero weight emits a
    ``RuntimeWarning`` and is silently clamped to 0.

    ``effective_lr`` is the η used for M_π; supplied by TrainEngine from
    the optimizer's current LR. Required (raises ``ValueError`` if
    missing or non-positive — M_π diverges at η=0).
    """
    if target_positions is None:
        raise RuntimeError(_MISSING_TARGETS_MSG)

    if weight != 0.0:
        warnings.warn(
            f"safety_monitor is observational; weight={weight!r} coerced to 0.0. "
            "Loss never contributes to gradient.",
            RuntimeWarning, stacklevel=2,
        )

    # Always-zero loss tensor that survives backward through any composition.
    # Multiply a learnable parameter's zero-of-itself so the autograd graph
    # survives, then scale to 0 — cleaner than a bare ``zeros(requires_grad)``
    # which breaks under some PyTorch versions when summed with graph-attached
    # tensors.
    device = next(model.parameters()).device
    zero_loss = torch.zeros((), device=device)

    B = len(samples)
    watch_sets = _resolve_watchlist(samples, watchlist, default_watchlist)

    # No positions anywhere ⇒ nothing to score; still emit the component
    # skeleton so downstream loggers don't see a missing-key flap.
    has_any_position = any(p for p in target_positions)
    if not has_any_position:
        return {
            "loss": zero_loss,
            "components": {
                "safety_monitor_eta": float(effective_lr or 0.0),
                "safety_monitor_alarm_count": 0,
                "safety_monitor_grower_size_mean": 0.0,
                "safety_monitor_grower_size_max": 0,
                "safety_monitor_M_p_mean": 0.0,
                "safety_monitor_M_p_min": 0.0,
                "safety_monitor_watchlist_hits": [],
            },
        }

    if effective_lr is None or effective_lr <= 0.0:
        raise ValueError(
            "safety_monitor requires a positive effective_lr for M_p(η); "
            "TrainEngine normally plumbs it from the optimizer's LR.",
        )
    if input_ids is None:
        raise RuntimeError(
            "safety_monitor requires input_ids from the main objective's "
            "forward batch. TrainEngine.step plumbs these automatically.",
        )

    with torch.no_grad():
        ids_dev = input_ids.to(device)
        attn_dev = attention_mask.to(device) if attention_mask is not None else None
        out = model(input_ids=ids_dev, attention_mask=attn_dev, use_cache=False)
        logits = out.logits.float()                                    # (B, T, V)

    M_p_vals: list[float] = []
    grower_sizes: list[int] = []
    watchlist_hits: list[tuple[int, int, int]] = []
    alarm_positions = 0

    for i, positions in enumerate(target_positions):
        if not positions:
            continue
        tokens = target_token_ids[i] if target_token_ids else []
        if len(tokens) != len(positions):
            raise ValueError(
                f"sample {i}: target_positions length {len(positions)} "
                f"does not match target_token_ids length {len(tokens)}",
            )
        for p, t in zip(positions, tokens):
            pi = F.softmax(logits[i, p], dim=-1)                       # (V,)
            mp = _m_p_value(pi, int(t), float(effective_lr))
            idx = torch.arange(pi.size(0), device=pi.device)
            grower_mask = (pi < mp) & (idx != int(t))
            grower = grower_mask.nonzero(as_tuple=True)[0].tolist()
            M_p_vals.append(mp)
            grower_sizes.append(len(grower))
            hits_i = [int(g) for g in grower if int(g) in watch_sets[i]]
            if hits_i:
                alarm_positions += 1
                for g in hits_i:
                    watchlist_hits.append((int(i), int(p), int(g)))

    mean_or_zero = lambda xs: (sum(xs) / len(xs)) if xs else 0.0       # noqa: E731
    components: dict[str, Any] = {
        "safety_monitor_eta": float(effective_lr),
        "safety_monitor_alarm_count": int(alarm_positions),
        "safety_monitor_grower_size_mean": float(mean_or_zero(grower_sizes)),
        "safety_monitor_grower_size_max": int(max(grower_sizes) if grower_sizes else 0),
        "safety_monitor_M_p_mean": float(mean_or_zero(M_p_vals)),
        "safety_monitor_M_p_min": float(min(M_p_vals) if M_p_vals else 0.0),
        "safety_monitor_watchlist_hits": watchlist_hits,
    }
    return {"loss": zero_loss, "components": components}
