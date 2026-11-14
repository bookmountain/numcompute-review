"""Execute every Quickstart code cell in one clean Python process.

This standard-library checker is useful in environments without Jupyter's
kernel tools. Cells share one namespace and run in notebook order. A failing
cell is reported and returns a nonzero exit status.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "demo" / "quickstart.ipynb"

_RUN_CELLS = r'''
import json
import sys
import traceback

cells = json.load(sys.stdin)
namespace = {"__name__": "__main__"}
for number, source in cells:
    try:
        exec(compile(source, f"notebook:cell-{number}", "exec"), namespace)
    except BaseException:
        print(f"Notebook code cell {number} failed:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
'''


def main(notebook_path=NOTEBOOK):
    """Run the notebook's code cells in a fresh Python process."""
    notebook_path = Path(notebook_path)
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Cannot read {notebook_path}: {error}", file=sys.stderr)
        return 1

    cells = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            cells.append((index, "".join(source) if isinstance(source, list) else source))
    if not cells:
        print("Notebook has no code cells to execute.", file=sys.stderr)
        return 1

    print(f"Executing {len(cells)} notebook code cells in a fresh Python process.")
    result = subprocess.run(
        [sys.executable, "-B", "-c", _RUN_CELLS],
        input=json.dumps(cells),
        text=True,
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        return result.returncode
    print("All notebook code cells passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else NOTEBOOK))
