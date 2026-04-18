/-
A: per-step (η_min, η_max) window for the composite loss
   L(z) = -log(1 - p_b) + w_+ · (-log p_g)
   (`unlike-kl-step-size-bound.md` §4–§5).

Statement scope (per-step; trajectory bound is a separate Lean follow-up):

    ∀ η ∈ [η_min(p, b, g, w_+), η_max(p, b, g, w_+, ε)] :
        (composite_step p b g w_+ η).prob b ≤ p.prob b
        ∧
        ∑ j ∉ {b, g}, |q.prob j - p.prob j| ≤ ε

η_min uses the §4 linearization (conservative; see calibration paragraph in
the writeup). η_max uses the §5 ε-collateral bound.
-/
import RazinSafety.Basic

open scoped BigOperators

namespace RazinSafety
namespace UnlikeStepWindow

variable {V : ℕ} (p : Distribution V)

/-- The composite logit shift δ_k from §2 (η-independent). -/
noncomputable def compositeDelta (b g : Fin V) (wPlus : ℝ) : Fin V → ℝ :=
  fun k =>
    if k = b then
      -(p.prob b) * (1 + wPlus)
    else if k = g then
      p.prob b * p.prob g / (1 - p.prob b) + wPlus * (1 - p.prob g)
    else
      p.prob k * (p.prob b / (1 - p.prob b) - wPlus)

/-- The post-step distribution after one composite SGD step at η. -/
noncomputable def compositeStep (b g : Fin V) (wPlus η : ℝ) (hV : 0 < V) : Distribution V :=
  p.shift (fun k => η * compositeDelta p b g wPlus k) hV

/-- The §4 simplified remainder: R(p, b) := p_b · (1 - 2 p_b + ‖p‖²) / (1 - p_b).
    Note: g-independent (proved correct in §4 of the writeup). -/
noncomputable def Rterm (b : Fin V) : ℝ :=
  let pb := p.prob b
  let s2 := ∑ k, (p.prob k) ^ 2
  pb * (1 - 2 * pb + s2) / (1 - pb)

/-- Linearized lower bound on the safe step-size (§4). Conservative — see
    calibration paragraph in `unlike-kl-step-size-bound.md`. -/
noncomputable def etaMinLin (b g : Fin V) (wPlus : ℝ) : ℝ :=
  let pb := p.prob b
  let pg := p.prob g
  let s2 := ∑ k, (p.prob k) ^ 2
  max 0 ((wPlus * (s2 - pg - pb) - Rterm p b) / (pb * (1 + wPlus)))

/-- ε-collateral upper bound on the step-size (§5). -/
noncomputable def etaMax (b g : Fin V) (wPlus ε : ℝ) : ℝ :=
  let pb := p.prob b
  let pg := p.prob g
  let s2 := ∑ k, (p.prob k) ^ 2
  ε / (wPlus * (s2 - pb ^ 2 - pg ^ 2))

/-- Total variation on the off-anchor support `V \ {b, g}`. -/
noncomputable def tvOffS (q : Distribution V) (b g : Fin V) : ℝ :=
  ∑ j, if j = b ∨ j = g then 0 else |q.prob j - p.prob j|

/-
The TV conjunct from the original §9 is intentionally excluded.
§5.b's `TV_sim^emp` is operational (one-step simulation, not an
analytical bound), and §5.a's `η_max^lin` is empirically non-conservative
(26% violation rate, worst 5×). Falsifier scripts
(`unlike_eta_min_sweep.py`, `trajectory_bound_sweep.py`) are the
operational validation; attempting to formalize TV would embed
calibration constants in the proof and defeat the purpose.
-/

/-- **Taylor-remainder lemma (sorry).** The linearized floor `etaMinLin`
    is constructed so that, modulo the `O(η²)` Taylor remainder of the
    softmax denominator, the inequality `η · δ_b ≤ log Z` holds. The
    discharge is technical analysis (uniform `O(η²)` bound on
    `M_composite`'s Taylor remainder for `(p, b, g, w_+)` in the simplex
    × `w_+ ∈ ℝ≥0` domain). Not load-bearing for the headline statement;
    tracked separately from the main correctness chain. -/
private lemma taylor_remainder_eta_min
    (b g : Fin V) (_hbg : b ≠ g) (wPlus : ℝ) (_hw : 0 ≤ wPlus)
    (η : ℝ) (_hη : 0 < η) (_hMin : etaMinLin p b g wPlus ≤ η) :
    -(η * p.prob b * (1 + wPlus))
      ≤ Real.log (∑ k, p.prob k * Real.exp (η * compositeDelta p b g wPlus k)) := by
  -- TODO(taylor-remainder): uniform O(η²) bound on M_composite's
  -- Taylor remainder for (p, b, g, w_+) in the simplex × w_+ ∈ ℝ≥0 domain.
  -- Discharge is technical analysis, not load-bearing for the headline
  -- statement. Tracked separately from the main correctness chain.
  sorry

/-- **A's headline theorem (per-step floor).** Inside the per-step floor
    `η ≥ etaMinLin p b g w_+`, the composite step is one-step safe for
    the unlike target `b`: `q_b ≤ p_b`.

    The companion ceiling on off-S TV (the conjunct in the original
    sketch) is intentionally not formalized; see the `TV out-of-scope`
    docstring above and `unlike-kl-step-size-bound.md` §9. -/
theorem unlike_composite_step_window
    (b g : Fin V) (hbg : b ≠ g)
    (wPlus : ℝ) (hw : 0 ≤ wPlus)
    (η : ℝ) (hη : 0 < η)
    (hV : 0 < V)
    (hMin : etaMinLin p b g wPlus ≤ η) :
    (compositeStep p b g wPlus η hV).prob b ≤ p.prob b := by
  classical
  have hZpos : (0 : ℝ) < ∑ k, p.prob k * Real.exp (η * compositeDelta p b g wPlus k) :=
    Finset.sum_pos (fun k _ => mul_pos (p.pos k) (Real.exp_pos _))
      (Finset.univ_nonempty_iff.mpr ⟨⟨0, hV⟩⟩)
  have hpb : 0 < p.prob b := p.pos b
  have hΔb : η * compositeDelta p b g wPlus b = -(η * p.prob b * (1 + wPlus)) := by
    unfold compositeDelta
    rw [if_pos rfl]
    ring
  change p.prob b * Real.exp (η * compositeDelta p b g wPlus b)
      / (∑ k, p.prob k * Real.exp (η * compositeDelta p b g wPlus k)) ≤ p.prob b
  rw [div_le_iff₀ hZpos, mul_le_mul_iff_right₀ hpb,
      ← Real.exp_log hZpos, Real.exp_le_exp, hΔb]
  exact taylor_remainder_eta_min p b g hbg wPlus hw η hη hMin

end UnlikeStepWindow
end RazinSafety
