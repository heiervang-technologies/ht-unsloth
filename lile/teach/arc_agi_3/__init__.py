"""ARC-AGI-3 benchmark adapter (Track D of the RLVR-online plan).

This package contains the *non-agentic* smoke path: load tasks, build a
prompt that shows train pairs, ask the student model for a single test
output grid, exact-match against ground truth. The agentic multi-turn
loop is layered on top in a follow-up.

Public surface:

- :func:`loader.load_tasks` — frozen task list (no network).
- :func:`prompts.build_prompt` — system+user prompt for a single task.
- :func:`runner.run_task` — async exact-match driver.
- ``verifier`` — registers ``"arc"`` on
  :mod:`lile.objectives.verifiers` so :func:`select` routes ARC-shaped
  prompts to the grid checker.
"""
