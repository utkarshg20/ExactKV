#!/usr/bin/env python3
"""Build Phase K launch pack artifacts (demo cards, launch manifest)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.launch_pack_builder import write_launch_pack  # noqa: E402


def main() -> int:
    paths = write_launch_pack(_ROOT)
    for label, path in paths.items():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
