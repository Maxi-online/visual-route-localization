"""Compile all source files without importing service runtimes."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
count=0
for p in root.rglob("*.py"):
    if any(x in {".git",".venv","node_modules","__pycache__"} for x in p.parts):continue
    compile(p.read_text(encoding="utf-8"),str(p),"exec");count+=1
for p in root.rglob("*.json"):
    if any(x in {".git",".venv","node_modules"} for x in p.parts):continue
    json.loads(p.read_text(encoding="utf-8"))
print(f"Compiled {count} Python modules and validated JSON files.")
