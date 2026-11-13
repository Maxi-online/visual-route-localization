"""Create missing local configuration from safe examples, without overwriting files."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
count=0
for p in root.rglob("*"):
    if any(x in {".git",".venv","node_modules","__pycache__"} for x in p.parts): continue
    if not p.is_file():continue
    if p.name==".env.example":name=".env"
    elif ".example." in p.name:name=p.name.replace(".example.",".")
    elif p.name.endswith(".json.example"):name=p.name.removesuffix(".example")
    else:continue
    target=p.with_name(name)
    if target.exists():continue
    shutil.copyfile(p,target);count+=1
    print(target.relative_to(root).as_posix())
print(f"Created {count} local configuration files; existing files preserved.")
