# lile glossary

Terms that appear in lile code comments and docs but are not standard
across the LLM finetuning literature. Cited when they first appear so
contributors can cross-reference.

---

## Razin-safe

An objective is **Razin-safe** if it cannot exhibit the likelihood-
displacement failure mode identified by Razin et al. (2024),
*"Unintentional Unalignment: Likelihood Displacement in Direct
Preference Optimization"*.

### The failure mode

Paired preference objectives (DPO, IPO, and close relatives) define a
gradient that rewards *raising the chosen log-prob relative to the
rejected*. Critically, "relative to" allows both to fall — as long as
the rejected falls further. When the chosen and rejected responses are
lexically or semantically similar, DPO can push the chosen *down* in
absolute terms while pushing the rejected down harder. The probability
mass shed by the chosen–rejected pair lands on some **third, unintended
output** that no gradient targeted. In practice the model is technically
"preferring chosen over rejected" while emitting neither.

### Why SFT-family objectives are safe

Pure SFT gradients are **likelihood-up on a concrete target string**.
No "push rejected down" term, no relative margin, nothing that can
fall. Anything the model now emits with higher probability came
directly from an example we actually wrote down.

### Quick reference per objective

| objective      | Razin-safe? | reason                                               |
|----------------|:-----------:|------------------------------------------------------|
| `sft`          | ✓           | likelihood-up on a target                            |
| `weighted_sft` | ✓           | likelihood-up (scaled per sample)                    |
| `ntp`          | ✓           | likelihood-up on raw text                            |
| `coh`          | ✓           | SFT on a hindsight trace (bad+critique+good)         |
| `kto`          | ✗ (mild)    | signed unary term — undesirable samples *are* pushed down; likelihood displacement bounded by the KL anchor but not excluded |
| `hinge`        | ✗           | pair margin                                          |
| `cppo`         | ✗           | multi-candidate preference                           |
| `ccpd_v2`      | ✗           | paired contrast with margin                          |

Not-Razin-safe does not mean *broken* — these objectives work and are
useful — only that they carry the theoretical risk and need safeguards
(KL anchors, reasonable pair sampling, reference-model divergence
caps). Razin-safe objectives give you a free pass on that concern.

### Why it matters for lile

lile's live-learning loop ingests small, noisy feedback batches and
applies them continuously. Any per-step likelihood displacement would
accumulate quickly with no batch-level averaging to smooth it out.
Razin-safety is therefore preferred as the *default* route for ingesting
user feedback — `nl_critique_with_rewrite` → `coh`, `rewrite` →
`weighted_sft`. Preference objectives are available (`hinge`, `cppo`,
`ccpd_v2`) but reserved for cases where the lesson genuinely is
*"this is better than that"* rather than *"this is good"*.

---

## References

- Razin, N., Malladi, S., Bhaskar, A., Chen, D., Saparov, A., Arora, S.
  (2024). *Unintentional Unalignment: Likelihood Displacement in Direct
  Preference Optimization*. NeurIPS 2024.
- Liu, H., Sferrazza, C., Abbeel, P. (2023). *Chain of Hindsight Aligns
  Language Models with Feedback*. (The Razin-safety of CoH is an
  explicit design goal — feedback is converted to a trace that is then
  SFT-trained.)
- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., Kiela, D.
  (2024). *KTO: Model Alignment as Prospect Theoretic Optimization*.
  (Unary, desirable/undesirable objective used in lile for binary
  thumbs up/down feedback.)
