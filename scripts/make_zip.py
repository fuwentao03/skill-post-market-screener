"""Package the skill as a zip for QuantSkills upload (方式一)."""
import zipfile
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.chdir(str(root))

exclude_dirs = {".git", "__pycache__", ".pytest_cache", "post_market_screener.egg-info",
                "cache", "output", ".idea", ".claude"}
exclude_files = {".env", ".gitignore"}

zip_path = root.parent / "skill-post-market-screener.zip"

with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
    count = 0
    for f in sorted(root.rglob("*")):
        if f.is_dir():
            continue
        parts = set(f.parts)
        if parts & exclude_dirs:
            continue
        if f.name in exclude_files:
            continue
        arcname = str(f.relative_to(root)).replace("\\", "/")
        zf.write(str(f), arcname)
        print(f"  + {arcname}")
        count += 1

size_mb = os.path.getsize(str(zip_path)) / (1024 * 1024)
print(f"\nDone: {count} files, {size_mb:.1f} MB")
print(f"Path: {zip_path}")
