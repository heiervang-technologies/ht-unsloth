"""Allow ``python -m lile <subcommand>`` execution."""

from __future__ import annotations

import sys

from lile.cli import main

if __name__ == "__main__":
    sys.exit(main())
