from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


_ARTIFACT_SUFFIXES = {".parquet", ".json", ".md", ".csv"}


def compare_timeaware_builds(first: Path, second: Path) -> dict[str, object]:
    first = Path(first).resolve()
    second = Path(second).resolve()
    first_files = _artifact_files(first)
    second_files = _artifact_files(second)
    missing_from_first = sorted(set(second_files) - set(first_files))
    missing_from_second = sorted(set(first_files) - set(second_files))
    mismatched = sorted(
        relative
        for relative in set(first_files) & set(second_files)
        if _sha256(first_files[relative]) != _sha256(second_files[relative])
    )
    status = (
        "ok"
        if not missing_from_first and not missing_from_second and not mismatched
        else "mismatch"
    )
    return {
        "status": status,
        "file_count": len(first_files) if status == "ok" else len(set(first_files) | set(second_files)),
        "missing_from_first": missing_from_first,
        "missing_from_second": missing_from_second,
        "mismatched_files": mismatched,
    }


def _artifact_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _ARTIFACT_SUFFIXES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two data_v4_timeaware builds")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    result = compare_timeaware_builds(args.first, args.second)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
