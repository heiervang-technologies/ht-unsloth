/-
B: SFT mass-flow characterization (`razin-safety-sharpened.md` §3).

Setup.  One SGD step on `L = -log p_t` with respect to logits `z = log p`,
step-size `η > 0`, gives
    Δ_k = η · (δ_{kt} - p_k)         (Kronecker delta at the target `t`).

Theorem (sft_mass_flow_iff).  For every non-target `j ≠ t`,
    q_j > p_j   ↔   p_j < M p t η
where
    M p t η  :=  -(1/η) · log (∑ k, p k * exp (η · (δ_{kt} - p k))).

The threshold `M p t η` is decreasing in η; `M p t 0+ = ‖p‖² - p_t` and
`M p t ∞ = p_t - 1`. The "unsafe regime" (some non-target grows) is at small η.
-/
import RazinSafety.Basic

open scoped BigOperators

namespace RazinSafety
namespace SftMassFlow

variable {V : ℕ} (p : Distribution V)

/-- The SFT logit shift at step-size η on target `t`. -/
noncomputable def sftShift (t : Fin V) (η : ℝ) : Fin V → ℝ :=
  fun k => η * ((if k = t then 1 else 0) - p.prob k)

/-- The post-step distribution `q` for one SFT SGD step. -/
noncomputable def sftStep (t : Fin V) (η : ℝ) (hV : 0 < V) : Distribution V :=
  p.shift (sftShift p t η) hV

/-- The per-token threshold `M_p(η)` defined in B §2. -/
noncomputable def threshold (t : Fin V) (η : ℝ) : ℝ :=
  -(1 / η) * Real.log (∑ k, p.prob k * Real.exp (η * ((if k = t then 1 else 0) - p.prob k)))

/-- **B's headline theorem.** For one SFT SGD step at η > 0,
    every non-target token `j` grows iff its prior probability is below the
    threshold `M_p(η)`. -/
theorem sft_mass_flow_iff
    (t : Fin V) (η : ℝ) (hη : 0 < η) (hV : 0 < V)
    (j : Fin V) (hjt : j ≠ t) :
    (sftStep p t η hV).prob j > p.prob j ↔ p.prob j < threshold p t η := by
  classical
  set Z : ℝ := ∑ k, p.prob k * Real.exp (η * ((if k = t then 1 else 0) - p.prob k))
    with hZdef
  have hZpos : 0 < Z :=
    Finset.sum_pos (fun k _ => mul_pos (p.pos k) (Real.exp_pos _))
      (Finset.univ_nonempty_iff.mpr ⟨⟨0, hV⟩⟩)
  have hpj : 0 < p.prob j := p.pos j
  have hηne : η ≠ 0 := ne_of_gt hη
  have hΔ : sftShift p t η j = -(η * p.prob j) := by
    unfold sftShift
    rw [if_neg hjt]
    ring
  have hq : (sftStep p t η hV).prob j
      = p.prob j * Real.exp (-(η * p.prob j)) / Z := by
    change p.prob j * Real.exp (sftShift p t η j) / Z = _
    rw [hΔ]
  rw [gt_iff_lt, hq]
  rw [lt_div_iff₀ hZpos, mul_lt_mul_iff_right₀ hpj]
  -- Goal: Z < exp(-(η * p_j)) ↔ p_j < threshold
  rw [← Real.exp_log hZpos, Real.exp_lt_exp]
  -- Goal: log Z < -(η * p_j) ↔ p_j < threshold
  unfold threshold
  have hrewr : -(1 / η) * Real.log Z = -Real.log Z / η := by field_simp
  rw [hrewr]
  constructor
  · intro h
    rw [lt_div_iff₀ hη]
    nlinarith
  · intro h
    rw [lt_div_iff₀ hη] at h
    nlinarith

end SftMassFlow
end RazinSafety
