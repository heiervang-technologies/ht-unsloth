/-
RazinSafety -- formalization of two results from `lile/docs/research/proofs/`:

  * `RazinSafety.SftMassFlow`     -- B: per-token characterization for one SFT step
                                     (`razin-safety-sharpened.md`)
  * `RazinSafety.UnlikeStepWindow` -- A: per-step (eta_min, eta_max) window for the
                                     composite unlike + w_+ * SFT-on-good loss
                                     (`unlike-kl-step-size-bound.md`)

  Both statements are per-step. The trajectory bound (composing N per-step audits)
  is a separate document and a separate Lean follow-up; see
  `unlike-trajectory-bound.md` (TBD).
-/

import RazinSafety.Basic
import RazinSafety.SftMassFlow
import RazinSafety.UnlikeStepWindow
