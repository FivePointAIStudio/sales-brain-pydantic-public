#!/usr/bin/env python3
"""Hash release source without including secrets, state or evaluator records."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
files=[]
for base in ["src","tests","docs","scripts"]:
 for path in sorted((root/base).rglob("*")):
  if not path.is_file() or any(x in {"__pycache__","runs","data","results"} for x in path.relative_to(root).parts) or path.name.startswith("._"):
   continue
  if path.suffix in {".pyc",".db",".sqlite",".sqlite3"}:continue
  files.append(path)
files += [root/x for x in ["pyproject.toml","README.md",".gitignore"] if (root/x).is_file()]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
blob=json.dumps(manifest,sort_keys=True,separators=(",",":"))
print(json.dumps({"source_sha256":hashlib.sha256(blob.encode()).hexdigest(),"files":manifest},indent=2))
