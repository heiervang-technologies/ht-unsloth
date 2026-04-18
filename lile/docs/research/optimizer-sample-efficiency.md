# lile — optimizer & sample-efficiency research

Status: draft. Owner: mei (%30, SVP of AI Research). Scope-partner: claude-opus (%4, replay/forgetting/per-sample slice + eval harness).

Seed issue: [#7 "lile: revisit optimizer choice for online-learning daemon"](https://github.com/heiervang-technologies/ht-unsloth/issues/7).

Dependency: [`eval-harness.md`](eval-harness.md) for the verifiable metric shape. All predictions in this doc are stated against that harness (HellaSwag `acc_norm`, ARC-Easy/Challenge `acc_norm`, GSM8K `exact_match`, HumanEval+ `pass@1`) at `n=100` per task.

**Harness framing (per PO, re: small-sample-reframe).** `n=100` yields ~±10pp CI half-width on binary accuracy, so the harness is a **regression check** — "does the model still have these skills after a streamed feedback phase?" — not a benchmark. Decision thresholds, applied uniformly to every optimizer A/B in this doc:

- **Pass**: no task drops more than **10pp** from baseline after streamed phase.
- **Catastrophic kill**: any task drops more than **20pp**.
- **Direction-only secondary**: mean across the four tasks does not drop.
- Improvements are observed but **not claimed as wins** at this `n`. The bar is skill preservation, not SOTA chasing.

This reframe boosts the case for per-objective param groups and ScheduleFree specifically via the **rehearsal-loop composition** argument; see §6.

---

## TL;DR

Two correctness PRs ship this week. Three optimizer A/Bs are research spikes gated on the harness landing. One family (Muon / Riemannion) is a defer-with-rationale for AdamW-pretrained Qwen3.

The single highest-leverage move is **per-objective param groups** — concern #2 (objective-mixing `v`-corruption) is the one most likely to suppress semantic-feedback uptake on a streaming daemon, because CoH / KTO / CCPD v2 gradient scales differ from SFT by up to ~10×, and the issue is hidden from binary-feedback-only evals. The second-highest is the **snapshot-load optimizer reset** — a tiny correctness bug that invalidates every A/B done around a snapshot boundary, which is how the PO's harness runs A/Bs.

The **rehearsal-loop** being built on top of the regression harness (see §6) additionally favors per-objective param groups and ScheduleFree-AdamW: both compose cleanly with bursty re-teach pulses in a way that a single shared `m`/`v` / constant-LR AdamW8bit does not.

Lion8bit is the cheapest structural alternative to per-objective groups (sign-based updates are natively scale-invariant per parameter) and should be benchmarked once the harness exists; it either replaces the per-group PR or complements it.

**Review flag for PR B — `bnb.optim.GlobalOptimManager` is a process-level singleton.** Multi-instance `AdamW8bit` over the same parameter tensors is undocumented and not a supported shape. Per-objective param groups under `AdamW8bit` would share the global registry while splitting state — no guaranteed correctness. PR B therefore switches the per-objective dict to `torch.optim.AdamW` (32-bit state) for the LoRA-only param set; memory cost is ~600 MB at 8B / ~50 MB at 0.6B, both rounding errors at our peak. Full rationale in §3.

---

## 1. Verdict per concern (issue #7 concerns 1–5)

Context: daemon target is 10–100 req/hr over days. At 100 req/hr × 24 h × ~2 train chunks/request = ~5k steps/day. A "session" is a few days to a few weeks between snapshot rotations.

### Concern 1 — Non-stationarity vs. Adam momentum

**Severity: LOW-MID.** AdamW's `β₂=0.999` decay constant τ≈1000 steps turns over once every ~5 hours under the target rate. Over a week-long session that's ~30 effective turnovers — more than enough for the EMAs to track non-stationary drift as long as no discontinuity (snapshot load, objective switch, replay burst) resets the distribution under `v`. The concern is not load-bearing on its own, **it is load-bearing via concerns #2 and #3**, which are the actual discontinuities.

Reproducer sketch: 2000-step stream with 30-step rolling-mean gradient cosine-similarity to first-step gradient. Expect cos > 0.3 throughout — slow drift, not non-stationarity in the adversarial sense.

### Concern 2 — Objective mixing corrupts `v` (the load-bearing one)

**Severity: HIGH** for semantic-feedback uptake; MASKED by `grad_clip=1.0` on binary-feedback-only streams.

Gradient magnitudes across registered objectives (`OBJECTIVES` in `lile/objectives/__init__.py`):
- **SFT / weighted_sft / NTP**: cross-entropy at per-token scale, O(1) per example.
- **KTO**: β=0.1 compresses log-ratio differences — per-sample gradient is ~0.1× SFT. Magnitude is further modulated by `λ_D=1.0, λ_U=1.5`.
- **CoH**: CE over critiqued→good transitions, ~SFT scale, but the signal concentrates on a fraction of target tokens.
- **Hinge**: margin-based; gradient is zero outside the margin, can spike at the margin edge.
- **CCPD v2**: composed of SFT + rank-advantage REINFORCE + KL; the REINFORCE component scales with advantage variance and is typically small.
- **KL anchor** (batch objective): `weight=0.1` in config → gradient ~0.1× base.

So between a CoH or SFT step and a KTO or KL-anchor step the gradient norm can differ by ~10×. After a long run of SFT-dominated steps, `v` ≈ ⟨g²⟩_SFT. When a KTO batch arrives, its effective LR is `lr · g_KTO / √v_SFT` — with `v_SFT ≈ (10 · g_KTO)²`, that's a ~10× LR boost on the KTO step. AdamW's `grad_clip=1.0` absorbs the worst of this at the norm level, but clipping doesn't restore the correct per-parameter direction.

**Why this bleeds into semantic feedback specifically.** The `nl_critique` and `nl_critique_with_rewrite` feedback paths route into CoH; `rewrite/preferred` routes into `weighted_sft`; binary `up/down` routes into KTO. So a stream that mixes binary and semantic feedback puts KTO, CoH, and weighted_sft into a single `v` accumulator. If the user thumbs-ups a lot and then writes one nl_critique, the CoH step sees a mis-scaled LR. If we only eval against binary feedback (pure KTO), concern #2 is invisible. **A metric that regresses on `nl_critique`-dominant streams but looks fine on binary-only is exactly the failure mode the PO called out.**

Reproducer (ships with the per-group PR's test):
```
1. Load Qwen3-0.6B + LoRA r=16, instantiate AdamW8bit.
2. Run 100 SFT steps on a fixed batch, log ||g||, record v-norm at the end.
3. Run 1 KTO step on a matched batch.
4. Assert: effective step ||Δθ_kto|| > k · ||Δθ_sft_avg|| for some k > 2.
   (Raw bnb does not expose v directly; infer via ||Δθ|| · √τ.)
5. Repeat with per-objective param groups: assert the anomaly disappears.
```

Fix: **per-objective param groups** (details in §3). Alternative structural fix: **Lion8bit** (sign-based = scale-invariant per parameter by construction; see §2 and §4).

### Concern 3 — Snapshot load does not reset optimizer state

**Severity: HIGH for correctness, MID for training quality.**

`Controller.request_snapshot_load` in `lile/controller.py:433-436` calls `snapshots.load(...)` via the queue but does not touch `TrainEngine._opt`. After a load, `m` and `v` are EMAs of gradients taken against weights that no longer exist — for ~τ_m ≈ 10 steps of `m` and ~τ_v ≈ 1000 steps of `v`, updates are subtly mis-scaled. Model recovers after ~10 steps, but those ~10 steps produce silently degraded deltas.

**Correctness impact bigger than quality impact.** The PO's A/B protocol explicitly loads snapshots to reset state between optimizer variants (`eval-harness.md` §A/B protocol). If `_opt` carries state across a snapshot-load boundary, *every A/B comparison is contaminated by residual momentum from the previous variant* — exactly the thing the protocol is designed to isolate. This is the same risk flagged as open-question #3 in `eval-harness.md`.

Fix is a one-liner in `_handle_task` at `controller.py:135-138`:
```python
elif kind == "snapshot_load":
    name = payload["name"]
    manifest = self.snapshots.load(name, self.state)
    self.train_engine._opt = None   # force rebuild on next step
    return {...}
```
Regression test: train N steps → save snapshot → train K more → load snapshot → assert `train_engine._opt is None and next step produces optimizer with step_count==0`.

### Concern 4 — Idle replay (T4.1) feeds the same optimizer

**Severity: LOW-MID.**

Idle replay fires at `idle_threshold_s=30` (config default), so under any realistic interactive load it adds ≤1–2 steps/hour. Replay uses the same `feedback_to_batch` routing (`controller.py:308-387`), so replayed records take the same objective path as their live counterparts. The failure mode is not replay-specific — it's that replay compounds concern #2 by doubling the hops between objectives in a single stream.

**Per-objective param groups makes this concern self-resolving.** Replay of a CoH record only touches the CoH optimizer state; live KTO steps don't see it in `v`. Without that change, severity hinges on how much mixing replay induces, which is workload-dependent and not worth chasing on its own.

### Concern 5 — Constant LR with no schedule

**Severity: LOW over days, MID over weeks.**

LoRA at `lr=1e-5` and `grad_clip=1.0` is conservative; the LoRA-as-regularizer finding ("LoRA Learns Less and Forgets Less") plus `merged_deltas` accumulation means we're already implicitly damping aggressive updates. The real concern is that we have no principled handle for "this session has been running for a week, should we decay or not?" — and neither constant-LR nor cosine-with-known-horizon fit that.

**ScheduleFree-AdamW is the right shape** — it's the only optimizer that explicitly targets no-known-horizon streaming ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682); empirical validation for streaming in [Baek et al., "Through the River", 2025](https://arxiv.org/abs/2507.09846)) — but the expected win is modest at LoRA rank 16 with conservative LR, and the `schedulefree` package is a real new dep. Defer to a research spike (§4).

---

## 2. Alternatives — quick assessment

| Optimizer | State | Addresses | Our fit | Ship class |
|---|---|---|---|---|
| AdamW8bit (current) | `m, v` | baseline | — | keep as fallback |
| **Per-objective AdamW** | `m, v` per obj | Concern #2 (directly) | **best structural fit** | ship-this-week (§3) |
| **Lion8bit** | `m` only | Concern #2 (structurally, via sign-invariance) + memory halved | cheapest drop-in alternative | **research-spike this week** (§4a) |
| ScheduleFree-AdamW | `m, v, z` | Concern #5 | narrow win, new dep | research-spike (§4b) |
| Muon / AdaMuon | `m` / `m, v` | optional concern #1 | **null on AdamW-pretrained Qwen3** per 2025 lit | **defer, cite** (§4c) |
| Riemannion-LoRA | Riemannian `m` | LoRA geometry | promising on Llama-3-8B commonsense; unproven on Qwen3 + code/math | defer behind Lion8bit A/B |
| Prodigy / Prodigy+SF | `m, v, s, d` | LR auto-tuning | overkill at LoRA r=16 | skip |
| Adafactor | factored `v` | memory | memory is not our bottleneck | skip |
| SGD+momentum | `m` | Concern #2 (no `v`) | needs hand-tuned per-obj LR anyway | skip |

### Lion8bit structural argument

Lion's update is `θ ← θ − lr · sign(β₁·m + (1−β₁)·g)`. The `sign` makes the per-parameter step magnitude exactly `lr` regardless of gradient scale — **concern #2 disappears by construction**. No param groups needed; one-line swap. 2025 benchmarks ([Chen et al., "Symbolic Discovery of Optimization Algorithms"](https://arxiv.org/abs/2302.06675); cautious-Lion in [Liang et al., 2024](https://arxiv.org/abs/2411.16085)) show Lion matches or beats AdamW on LM pretraining at half the optimizer-state memory. The known hazards (non-convergence from the sign discretization; [RLion paper, 2025](https://www.nature.com/articles/s41598-025-07112-4)) show up in specific settings — worth watching HumanEval+ and GSM8K for variance, not a blocker.

### Muon / Riemannion call for Qwen3

The Moonlight paper ([Liu et al., 2025](https://arxiv.org/html/2502.16982v1)) ran the exact experiment: **SFT with Muon does not beat AdamW when pretrain optimizer differs**. Confirmed independently on Llama-3.2-3B in [Shi et al., "Effective Quantization of Muon Optimizer States", 2025](https://arxiv.org/html/2509.23106v1) and on Qwen2-0.5B / SmolLM2-360M / GPT2-medium in [MuonAll, Nov 2025](https://arxiv.org/html/2511.06086v1). Qwen3 is AdamW-pretrained. The empirical ceiling on a Muon A/B for us is "matches AdamW" — that is not a win under the PO's eval harness bar.

Riemannion-LoRA ([Bogachev et al., 2025](https://arxiv.org/abs/2507.12142)) reports consistent commonsense-reasoning wins on Llama-3-8B at LoRA r=16, but **does not evaluate on GSM8K or HumanEval**. At r=16 the paper reports up to 32.5% wall-clock overhead vs Adam. For a daemon whose whole point is low-latency online updates, that is a meaningful tax. Defer until Lion8bit A/B lands and we know whether the scale-invariance angle alone captures most of the structural win.

---

## 3. Per-objective param groups — minimal diff sketch

### The bitsandbytes quirk

`bnb.optim.AdamW8bit` registers each param into the process-global `GlobalOptimManager` singleton (see [bnb optimizer docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)). The manager holds per-parameter overrides keyed by `id(param)`. Instantiating **two AdamW8bit optimizers over the same parameter tensor** is not a supported shape — the two `Optimizer8bit.state` dicts will each hold their own `m`/`v` quantization state, but the `GlobalOptimManager` registration is shared, and the 8-bit dequant/requant path is not designed for that. No crash guaranteed, but no correctness guaranteed either.

**Recommendation: switch the per-objective optimizer set to plain `torch.optim.AdamW`, not AdamW8bit.** At LoRA r=16 on Qwen3-8B there are ~12.8M trainable params → 102 MB of 32-bit optimizer state per objective × 6 registered objectives ≈ 600 MB. We measured peak 8.21 GB on the §11 bench at 8B; adding 600 MB is a rounding error. For Qwen3-0.6B it's ~50 MB total. **Memory is not the bottleneck; correctness is.**

Keep AdamW8bit as the single-optimizer fallback when `cfg.per_objective_optim=False` (the default, for one release cycle).

### The diff (approximate, ~80 LOC)

```python
# lile/engine/train.py

class TrainEngine:
    def __init__(self, state, lr=1e-5, grad_clip=1.0, per_objective=False):
        self.state = state
        self.lr = lr
        self.grad_clip = grad_clip
        self.per_objective = per_objective
        self._opts: dict[str, torch.optim.Optimizer] = {}  # keyed by objective name; "" = shared

    def _optimizer(self, objective: str) -> torch.optim.Optimizer:
        key = objective if self.per_objective else ""
        if key not in self._opts:
            params = [p for p in self.state.model.parameters() if p.requires_grad]
            if self.per_objective:
                self._opts[key] = torch.optim.AdamW(params, lr=self.lr)  # plain torch — bnb isn't safe across per-key instances
                log.info("per-objective torch.AdamW created for %r (lr=%g)", key, self.lr)
            else:
                try:
                    import bitsandbytes as bnb
                    self._opts[key] = bnb.optim.AdamW8bit(params, lr=self.lr)
                except Exception:
                    self._opts[key] = torch.optim.AdamW(params, lr=self.lr)
        return self._opts[key]

    def step(self, spec):
        ...
        opt = self._optimizer(name)   # name already in scope at line 66
        opt.zero_grad()
        loss.backward()
        ...
        opt.step()
```

And in `Controller.request_snapshot_load` — couples with concern #3 fix:
```python
elif kind == "snapshot_load":
    ...
    self.train_engine._opts.clear()   # force all per-objective optimizers to rebuild
```

### Regression test shape

`lile/tests/test_per_objective_param_groups.py`:
1. Instantiate TrainEngine with `per_objective=True`.
2. Run 50 SFT steps, then 5 KTO steps.
3. Assert `engine._opts["sft"].state_dict()["state"]` has step-count ≥ 50; `engine._opts["kto"]` has step-count ≥ 5.
4. Run the reproducer from §1 concern #2: measure `||Δθ||` ratio; assert it does not exhibit the >2× spike that the single-optimizer case exhibits.
5. Smoke test: SFT loss still descends on Qwen3-0.6B (same shape as `smoke_objectives.py`).

### Gotchas

- The **CCPD v2** objective calls `FastLanguageModel.for_inference()` unconditionally inside `_sample_candidates` (see `STATUS.md` "Known caveats"). That flip is safe because `ModelState.mode_lock` is held; the optimizer swap does not change that assumption.
- The **KL anchor** is a batch-level objective that composes onto the primary. Under per-group, should KL anchor go in the primary objective's group (with its own scale folded in) or its own group? **Recommendation: folded into primary.** The anchor's gradient is by design correlated with the primary's, and isolating it would break the KL-LR coupling users expect.
- Replay steps (`spec["_replay"]=True`) should use the same per-objective optimizer as their live counterparts. This is automatic if we key on `spec["objective"]`, which is what idle replay populates.

---

## 4. Prioritized PR list

All PRs target `ht` branch; squash-merge per repo convention.

All eval gates below are stated against the `n=100` regression-check framing from §TL;DR: **pass iff no task drops >10pp**; catastrophic kill at >20pp; direction-only mean-across-tasks secondary. Improvement pps are not claimed at this `n`.

| # | PR | LOC (approx) | Class | Eval gate | Expected signal (qualitative) |
|---|---|---|---|---|---|
| **A** | Reset optimizer state on snapshot load (concern #3) | ~25 + 1 test | **land this week** | none — pure correctness; harness-independent | Protects every subsequent A/B from ghost-momentum contamination; reduces run-to-run variance on A/Bs that cross a snapshot boundary |
| **B** | Per-objective param groups behind `cfg.per_objective_optim` flag (concern #2) | ~80 + 1 test | **land this week** (after harness + baseline land) | **must pass gate on a mixed stream with ≥30% `nl_critique` events** | Baseline AdamW8bit is expected to show larger eval drops on semantic-heavy streams than on binary-only; PR B should narrow that gap. Composes cleanly with rehearsal pulses (§6) — each skill's rehearsal batch hits its own `m`/`v`. |
| **C** | Lion8bit A/B behind `cfg.optimizer="lion8bit"` | ~30 + bench | **research spike** | **must pass gate on all four tasks on the standard 500-event stream** | Sign-invariance removes the concern #2 failure mode by construction, same effect as PR B without the per-objective plumbing. Worth running to see if PR B is redundant. |
| **D** | ScheduleFree-AdamW A/B | ~40 + bench + `schedulefree` dep | **research spike** | **must pass gate on a 2000-event stream (4× target daily volume)**; secondary test on a stream with rehearsal pulses | Expected comparable to AdamW on the standard stream; differentiates on long streams and under rehearsal, where "no-known-horizon" is the native shape (§6). |
| **E** | Muon / Riemannion A/B | ~120 + bench + impl | **defer** | would need to pass on all four tasks while being non-trivial wall-clock overhead | Lit predicts null effect on AdamW-pretrained bases. Not worth the spike budget until independent Qwen3 replication exists. |
| **F** | Idle-replay per-objective isolation | subsumed by B | — | subsumed | — |

**Shortlist for Friday merge**: A, and either B or C depending on what lands in the harness side.

A is risk-free correctness and ships regardless. For the second slot:

- If the rehearsal loop lands before Friday → run **C (Lion8bit)** first; if Lion8bit passes gate cleanly on the rehearsal-enabled harness, it subsumes B and ships as the new default.
- If the rehearsal loop does not land before Friday → ship **B** behind the flag; it's the more conservative fix and doesn't commit us to a new optimizer class.
- In either case, the other spike (B or C) stays on the board for next week.

The one outcome to avoid: shipping B and C in the same week. Each needs its own clean A/B against a stable baseline; stacking them makes attribution impossible on an `n=100` regression harness.

### Falsifiable regression criteria against the eval harness

Stated against `eval-harness.md`'s `n=100` regression-check framing. "Pass" = no task drops >10pp from baseline; "catastrophic" = any task drops >20pp; "direction-only" = mean-across-tasks secondary.

- **PR A**: No direct eval prediction — this is a correctness bug fix. Verification is indirect: the harness's inter-run variance on A/Bs that cross a snapshot-load boundary should drop after this PR. Measure once (3 seeds of the same stream, pre/post), confirm variance reduction is qualitative, do not gate on it.
- **PR B**: Baseline AdamW8bit on a mixed-feedback stream (at least 30% `nl_critique`) is expected to show one-or-more task regressions closer to the 10pp gate than on a binary-only stream. PR B should shift both profiles to "all four tasks pass gate cleanly". Catastrophic kill: if PR B causes any task to drop >20pp on either stream. Direction-only secondary: mean across tasks after PR B ≥ mean after baseline.
- **PR C**: Lion8bit on the standard 500-event stream must pass gate on all four tasks. Secondary: if Lion8bit passes where baseline AdamW8bit fails (on the same stream), that's evidence the scale-invariance fix (§2) is real and PR B is redundant. Kill if Lion8bit fails gate on any task or produces catastrophic regression.
- **PR D**: ScheduleFree-AdamW on the standard 500-event stream must pass gate — same as any other A/B. Discriminator: on a 2000-event stream, ScheduleFree should pass gate where constant-LR AdamW is **more likely** to have drifted close to a task's 10pp threshold. Kill if no discriminator shows up by 2000 events.
- **All PRs (composition with rehearsal loop, §6)**: on any stream where a rehearsal pulse fires mid-stream, PR B and PR D should show smoother post-rehearsal eval trajectories than baseline. Exact instrumentation TBD once the rehearsal loop is wired — the composition argument is the doc-level payoff; the on-eval signal is a nice-to-have.

### CI baseline choice

Aligned with `eval-harness.md` §Baseline choice:
- **Per-PR CI**: Qwen3-0.6B at `--limit 100` on all four tasks (~15 min).
- **Maintainer-triggered pre-merge**: Qwen3-8B at `--limit 250` (~30 min).
- **Weekly regression**: Full evals (`--limit` unset) at 0.6B and 8B on `ht`.

0.6B-only CI is defensible because `bench_rc_ranking.py` already established that at this scale, ranking-reliability patterns track between 0.6B and 8B at matched `k` (see `STATUS.md` §11 benchmark decision). If we see a 0.6B/8B divergence on a given PR, that is itself signal worth investigating, not a reason to skip 0.6B.

---

## 5. Experiments we should NOT run (and why)

| Experiment | Why not |
|---|---|
| **Muon from scratch for SFT on Qwen3** | Three 2025 papers (Moonlight, Effective-Quantization-Muon, MuonAll) converge on "null effect when pretrain optimizer differs." Qwen3 is AdamW-pretrained. Expected ceiling = AdamW parity, which fails the PO's "must move a metric by ≥1pp" bar. |
| **Adafactor on LoRA** | Factored `v` saves tens of MB at our scale. We are not memory-bound (8.21 GB peak on 8B; 4090/5090 headroom plentiful). Its relative-step schedule is the opposite of what concern #5 asks for. |
| **Prodigy / Prodigy+SF** | Designed for finite-epoch "escape-velocity" LR finding. Our scenario is infinite streams. Mis-match in optimization primitive — would need significant adaptation, and the per-parameter state overhead (16 B/param) is wasted at LoRA scale. |
| **SGD + momentum with hand-tuned per-objective LR** | The per-objective param-group PR gives us per-objective `m`/`v` with an AdamW framing we already understand. SGD would require re-learning the LoRA LR landscape; no obvious reason to. |
| **Riemannion-LoRA before Lion8bit A/B lands** | Riemannion's 32.5% wall-clock overhead at r=16 is the kind of cost only justified by large wins on metrics we care about. The published paper covers commonsense but not code/math. Spike it only if Lion8bit A/B leaves unexplained gaps on HellaSwag/ARC. |
| **Full optimizer replacement in a single PR** | Two-concern PRs are for bug fixes, not experiments. Each optimizer gets its own flag, its own A/B, its own merge. |
| **Auto-LR like LR-range-test on streaming data** | Requires a bounded test set with known baseline; we're online. Ship ScheduleFree (D) if we want LR-freedom. |

---

## 6. Interaction with semantic feedback & the rehearsal loop

The PO flagged two framings that both pull toward the same conclusion:

1. A win on binary-feedback-only evals that regresses semantic-feedback uptake is not a win.
2. The harness is being extended into a **skill-targeted re-teach loop** — on regression detection, a small canonical rehearsal batch (`lile/teach/rehearsal/<task>.jsonl`, ~50 samples per skill) gets pushed through `/v1/train` and the model is re-evaluated. This converts the harness from a gate into a closed-loop rehearsal mechanism.

### Per-feedback-kind mapping

Mapping each PR to the feedback types they touch (from `Controller.feedback_to_batch`, `controller.py:308-387`):

| Feedback kind | Objective hit | PR A (snapshot reset) | PR B (per-obj groups) | PR C (Lion8bit) | PR D (ScheduleFree) |
|---|---|---|---|---|---|
| `binary` (up/down) | KTO | no change | **isolates KTO `v` from SFT history** — predicted win | scale-invariance removes the concern structurally | schedule-free is orthogonal; no direct interaction |
| `rewrite` / `preferred` | weighted_sft | no change | **isolates from CoH/KTO** — predicted win on rewrite-heavy streams | scale-invariance helps; small effect | orthogonal |
| `nl_critique` | CoH | no change | **biggest isolation win** — CoH scale differs from SFT on target-token fraction | scale-invariance helps the most here | orthogonal |
| `nl_critique_with_rewrite` | CoH (good-path) | no change | same as `nl_critique` | same | orthogonal |
| (opt-in) CCPD v2 | CCPD composite | no change | depends on whether CCPD's internal SFT/REINFORCE/KL share a group — recommend **one group per objective name**, not per component | scale-invariance per-step is fine; CCPD's τ-spread skip handles discriminative failures orthogonally | orthogonal |

The per-objective PR (B) is the one most likely to measurably improve semantic-feedback uptake. This is also the hypothesis that the PO's harness `streamed.json` can falsify cleanly — run a 100-nl_critique stream, compare adapter eval before and after, PR B should show post-stream eval closer to baseline than baseline-optimizer does.

### Composition with the rehearsal loop

The rehearsal loop fires **bursty** training pulses on a mix of objectives whenever a regression is detected. Each pulse is a small batch (~50 samples, typically SFT-shaped for the target skill), interleaved with whatever live feedback is arriving. The optimizer's job is to make the pulse's gradient count, without either overfitting to the rehearsal batch or having the rehearsal batch derailed by stale statistics from the preceding live stream.

How each PR composes with this loop:

- **PR A (snapshot-load optimizer reset).** Orthogonal — rehearsal doesn't load snapshots by default. But if the rehearsal loop ever does a "reset to last-known-good snapshot → pulse → eval" (a plausible extension), PR A is a correctness prerequisite. Mark as a hard dependency for any rehearsal implementation that uses snapshots.
- **PR B (per-objective param groups).** *Strong compositional fit.* A rehearsal pulse for the HellaSwag skill is a run of SFT samples. Under single-optimizer AdamW, that SFT pulse competes with the live stream's `v` — if the stream has been KTO-dominant, the SFT pulse's effective LR gets inflated (concern #2 in reverse). Per-objective groups give the rehearsal SFT pulse its own clean `v` history. **Prediction: PR B + rehearsal loop is more stable than baseline + rehearsal loop on streams with high SFT/KTO mixing.**
- **PR C (Lion8bit).** Same compositional story as PR B via the sign-invariance route. Rehearsal pulses don't suffer from mixed-objective `v` corruption because Lion has no `v`. If Lion8bit ships, PR B's compositional argument is subsumed.
- **PR D (ScheduleFree-AdamW).** *Natural fit.* ScheduleFree has no pre-specified horizon; a rehearsal pulse doesn't need to "rewind" or re-tune LR. The schedule is always local. Under a cosine or linear-decay schedule, a rehearsal pulse dropped mid-way into a schedule gets a degraded LR that was tuned for the live stream, not for the pulse. ScheduleFree is the only optimizer in this set that makes "live stream + rehearsal pulse" a single coherent optimization process. **This moves PR D up from "research spike with modest value" to "research spike with a direct compositional argument once rehearsal lands".**
- **PR E (Muon / Riemannion).** No specific compositional argument beyond Lion's. Defer stands.

### What this changes in the PR priority

- PR B's value grows — add "rehearsal composition" to its motivation.
- PR D's value grows — no longer contingent purely on long-stream drift; rehearsal makes it useful on shorter streams too.
- PR C's value unchanged but its relationship to B changes: if Lion8bit wins under rehearsal, it subsumes B. Run Lion8bit A/B first if time is tight.

---

## 7. Resolved threads with the PO (`%4`)

All six open threads were resolved in PO ping `re: six-open-threads`. Summarized here as decisions-of-record for downstream readers. Companion synthesis at [`sample-efficiency-synthesis.md`](sample-efficiency-synthesis.md) widens the frame to the three failure modes (forgetting, optimizer-mixing, input signal loss), ranks forgetting above optimizer-mixing in expected severity, and proposes two harness-independent companions to this slice: PR G (full-sequence KL anchor, ~15 LOC) and PR H (rehearsal loop, ~200 LOC). An EMA-normalized per-objective loss coefficient (~20 LOC, documented in synthesis §3) is a noted fallback if PR B's per-objective `v` split fails its falsifiable criterion — it targets the same gradient-scale spread from the loss side rather than the optimizer side, and composes cleanly with AdamW8bit so it does not collide with the GlobalOptimManager singleton.

1. **Eval harness landing order — resolved.** PR A ships Friday unconditionally. PR B holds for harness + baselines; slides a week if harness slips past Thursday. Lion8bit / ScheduleFree spikes gated on harness by construction.
2. **CCPD v2 param-group — resolved: one group for the whole objective.** Hard-coded in PR B. Rationale: CCPD v2's internal component loss magnitudes are already within-scale; splitting adds state memory with no `v`-corruption win. Re-openable if later measurement shows internal variance swamps across-objective variance.
3. **`mixed_500.jsonl` composition — resolved.** Authoritative distribution (owner: PO, ETA Thursday):
   - **By kind:** 40% binary (KTO) · 30% `nl_critique` (CoH / CCPD v2) · 20% `rewrite` (`weighted_sft` w=3.0) · 10% `preferred` (hinge)
   - **By domain:** 25% math · 25% code · 25% common-sense · 25% general/physics/factual
   - **Ordering:** shuffled with fixed seed 42; not domain-clustered
   - **Sources:** GSM8K train (math), HumanEval train (code), HellaSwag train (common-sense), MMLU diverse (general); responses synthesized by the cold model; labels from ground-truth where verifiable, NL critiques hand-written or LLM-synthesized against a fixed rubric

   30% `nl_critique` comfortably exercises concern #2; PR B's falsifiable-regression test uses this stream verbatim.
4. **8B maintainer-triggered eval — resolved.** CI on 0.6B always. Pre-merge 8B block only for PRs touching `lile/engine/`, `lile/objectives/`, `lile/state.py`, `lile/controller.py`. Codify as a GitHub Actions path filter when wiring CI. PR B, C, and D all fall under the 8B block; PR A does too (touches `controller.py`).
5. **Replay/forgetting — resolved.** Each replay step uses the same objective key as its live origin; PR B's per-objective dict sees replay and live as the same key. `_replay_multiplier` weighting is out of scope for this research phase (potential PR F, orthogonal to optimizer choice).
6. **Rehearsal-loop trigger — resolved: CI-driven only.** Harness CI detects regression → opens issue or emits GH Actions artifact. Daemon-side watcher + auto-rehearsal is a separate production PR, handed to the backend track (`%32`), not this slice. PR D's compositional case is testable against CI-driven rehearsal alone — no daemon-side dependency.

---

## References

- [Issue #7 — lile: revisit optimizer choice for online-learning daemon](https://github.com/heiervang-technologies/ht-unsloth/issues/7) (seed)
- [`lile/docs/research/eval-harness.md`](eval-harness.md) (PO's eval slice)
- [`lile/docs/research/sample-efficiency-synthesis.md`](sample-efficiency-synthesis.md) (PO's synthesis across all three failure modes)
- [`lile/STATUS.md`](../../STATUS.md) and [`lile/DESIGN.md`](../../DESIGN.md) (current state + invariants)
- Liu et al., "Muon is Scalable for LLM Training" (Moonlight), [arXiv:2502.16982](https://arxiv.org/html/2502.16982v1)
- Shi et al., "Effective Quantization of Muon Optimizer States", [arXiv:2509.23106](https://arxiv.org/html/2509.23106v1)
- Parakh et al., "MuonAll: Muon Variant for Efficient Finetuning of LLMs", [arXiv:2511.06086](https://arxiv.org/html/2511.06086v1)
- Bogachev et al., "LoRA meets Riemannion", [arXiv:2507.12142](https://arxiv.org/abs/2507.12142)
- Defazio et al., "The Road Less Scheduled" (Schedule-Free), [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- Baek et al., "Through the River", [arXiv:2507.09846](https://arxiv.org/abs/2507.09846)
- Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion), [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Liang et al., "Cautious Optimizers: Improving Training with One Line of Code", [arXiv:2411.16085](https://arxiv.org/abs/2411.16085)
- bitsandbytes 8-bit optimizers: [HF docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
- `schedulefree` PyPI package: [pypi.org/project/schedulefree](https://pypi.org/project/schedulefree/)
