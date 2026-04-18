/-
Shared definitions: discrete probability simplex on `Fin V`, softmax, and the
two one-step updates used downstream.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Field

open scoped BigOperators

namespace RazinSafety

variable {V : ℕ}

/-- A point on the open probability simplex on `Fin V`. -/
structure Distribution (V : ℕ) where
  prob : Fin V → ℝ
  pos  : ∀ i, 0 < prob i
  sum_one : ∑ i, prob i = 1

namespace Distribution

variable (p : Distribution V)

/-- Softmax of `z : Fin V → ℝ`. Returns a `Distribution`. -/
noncomputable def softmax (z : Fin V → ℝ) (hV : 0 < V) : Distribution V := by
  classical
  have hpos : (0 : ℝ) < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _)
      (Finset.univ_nonempty_iff.mpr ⟨⟨0, hV⟩⟩)
  refine ⟨fun i => Real.exp (z i) / ∑ j, Real.exp (z j), ?_, ?_⟩
  · intro i; exact div_pos (Real.exp_pos _) hpos
  · rw [← Finset.sum_div, div_self (ne_of_gt hpos)]

/-- The pre-step logit-space distribution `q` after additive logit shift `Δ`:
    `q_k = p_k * exp Δ_k / Z`. This is the parameterization used in both proofs. -/
noncomputable def shift (Δ : Fin V → ℝ) (hV : 0 < V) : Distribution V := by
  classical
  have hZ : (0 : ℝ) < ∑ k, p.prob k * Real.exp (Δ k) :=
    Finset.sum_pos (fun k _ => mul_pos (p.pos k) (Real.exp_pos _))
      (Finset.univ_nonempty_iff.mpr ⟨⟨0, hV⟩⟩)
  refine ⟨fun k => p.prob k * Real.exp (Δ k) / ∑ j, p.prob j * Real.exp (Δ j), ?_, ?_⟩
  · intro k; exact div_pos (mul_pos (p.pos k) (Real.exp_pos _)) hZ
  · rw [← Finset.sum_div, div_self (ne_of_gt hZ)]

end Distribution
end RazinSafety
