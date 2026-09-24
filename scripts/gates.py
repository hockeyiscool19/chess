"""Run every landing gate: line budget, Ruff (lint and format), Pyright, and pytest.

    uv run python scripts/gates.py

These mirror the autoresearcher monorepo's compliance gate so this repository
lands under the same rules when it is used as the monorepo's ``apps/chess``.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_BUDGET = 400
CHECKED = ("src", "tests", "scripts")
SUFFIXES = (".py", ".js", ".css", ".html")
STEPS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", *CHECKED),
    ("ruff", "format", "--check", *CHECKED),
    ("pyright",),
    ("pytest", "-q"),
)


def over_budget() -> list[str]:
    """Return files longer than the line budget."""
    found: list[str] = []
    for folder in CHECKED:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                lines = len(path.read_text(encoding="utf-8").splitlines())
                if lines > LINE_BUDGET:
                    found.append(f"{path.relative_to(ROOT)}: {lines} lines > {LINE_BUDGET}")
    return found


def main() -> int:
    """Run the gates in order and return non-zero if any fails."""
    failed = 0
    for message in over_budget():
        print(message)
        failed = 1
    for step in STEPS:
        executable = shutil.which(step[0])
        if executable is None:
            print(f"{step[0]} is not installed; run `uv sync` first")
            return 1
        print(f":: {' '.join(step)}", flush=True)
        completed = subprocess.run([executable, *step[1:]], cwd=ROOT, check=False)  # noqa: S603
        failed = failed or completed.returncode
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
