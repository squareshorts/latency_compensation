import csv
from pathlib import Path

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "docs" / "pre_pivot_file_inventory.csv"


def category(path: Path) -> str:
    rel = path.relative_to(ROOT)
    first = rel.parts[0] if rel.parts else ""
    return {
        "data": "raw_or_extracted_data",
        "downloads": "archive",
        "external": "external_dependency",
        ".venv-not-robotics": "python_environment",
        "results": "derived_result",
        "scripts": "script",
        "docs": "documentation",
    }.get(first, "repository_root")


def main() -> None:
    files = sorted((p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts), key=lambda p: str(p).lower())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_name(OUT.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "relative_path", "category", "size_bytes", "modified_local", "large_file_ge_100mb"])
        writer.writeheader()
        for path in files:
            stat = path.stat()
            writer.writerow({
                "path": str(path),
                "relative_path": str(path.relative_to(ROOT)),
                "category": category(path),
                "size_bytes": stat.st_size,
                "modified_local": stat.st_mtime,
                "large_file_ge_100mb": stat.st_size >= 100 * 1024 * 1024,
            })
    temp.replace(OUT)
    print(f"Inventoried {len(files)} files")


if __name__ == "__main__":
    main()
