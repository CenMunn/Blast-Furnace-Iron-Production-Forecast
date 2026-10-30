from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


_INCLUDED_SUFFIXES = {".parquet", ".npy", ".json", ".md", ".txt"}
_EXCLUDED = {"manifests/development_log.md"}


def compare_builds(first: Path, second: Path) -> dict[str, Any]:
    first = Path(first).resolve()
    second = Path(second).resolve()
    first_hashes = _artifact_hashes(first)
    second_hashes = _artifact_hashes(second)
    missing_from_second = sorted(set(first_hashes) - set(second_hashes))
    extra_in_second = sorted(set(second_hashes) - set(first_hashes))
    changed = sorted(
        relative
        for relative in set(first_hashes) & set(second_hashes)
        if first_hashes[relative] != second_hashes[relative]
    )
    status = "ok" if not (missing_from_second or extra_in_second or changed) else "mismatch"
    return {
        "status": status,
        "compared_artifacts": len(set(first_hashes) & set(second_hashes)),
        "missing_from_second": missing_from_second,
        "extra_in_second": extra_in_second,
        "changed": changed,
        "first_hashes": first_hashes,
        "second_hashes": second_hashes,
    }


def audit_reproducibility(source_root: Path, delivery_root: Path) -> dict[str, Any]:
    from src.build_data_v3 import build_data_v3

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "build_1"
        second = temporary_root / "build_2"
        build_data_v3(source_root, first)
        build_data_v3(source_root, second)
        rebuilt = compare_builds(first, second)
        delivered = compare_builds(first, delivery_root)
        status = "ok" if rebuilt["status"] == delivered["status"] == "ok" else "mismatch"
        return {"status": status, "two_fresh_builds": rebuilt, "fresh_vs_delivery": delivered}


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in _INCLUDED_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in _EXCLUDED:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="data_v3 可复现性审查")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("delivery_root", type=Path)
    arguments = parser.parse_args()
    result = audit_reproducibility(**vars(arguments))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
