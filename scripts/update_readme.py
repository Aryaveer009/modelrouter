"""Copy the headline results table from results/results.md into README.md.

Keeps README numbers identical to the benchmark output instead of hand-typed.
Run: python scripts/update_readme.py

Author: Aryaveer Mankotia (https://github.com/Aryaveer009)
"""

import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
results = (root / "results" / "results.md").read_text(encoding="utf-8")
lines = results.splitlines()
start = next(i for i, ln in enumerate(lines) if ln.startswith("| Strategy"))
end = start
while end < len(lines) and lines[end].startswith("|"):
    end += 1
table = "\n".join(lines[start:end])

readme_path = root / "README.md"
readme = readme_path.read_text(encoding="utf-8")
block = f"<!-- RESULTS:START -->\n{table}\n<!-- RESULTS:END -->"
new, n = re.subn(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", block, readme, flags=re.S)
assert n == 1, "README is missing the RESULTS markers"
readme_path.write_text(new, encoding="utf-8")
print("README results table updated")
