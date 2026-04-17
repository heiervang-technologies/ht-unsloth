"""lile — live-learning local LLM daemon. See lile/PLAN.md for the spec."""

__version__ = "0.1.0-dev"


def install() -> None:
    """Install the matmul_lora residual patch. Idempotent. Called automatically
    by ``ModelState.load``; expose here for callers that want to pre-warm
    before constructing a ModelState (e.g. offline snapshot inspection that
    subsequently instantiates the model)."""
    from .state import install as _install
    _install()
