"""Set OPENROUTER_MODEL in .env (UTF-8 no BOM)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    lines = ENV.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith("OPENROUTER_MODEL="):
            out.append(f"OPENROUTER_MODEL={args.model}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"OPENROUTER_MODEL={args.model}")
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"OPENROUTER_MODEL={args.model}")


if __name__ == "__main__":
    main()
