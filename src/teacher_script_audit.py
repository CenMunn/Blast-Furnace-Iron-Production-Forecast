from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_REQUESTED_FILE_NAMES = (
    "teacher_transformer.py",
    "teacher_model.py",
    "老师脚本.py",
    "老师Transformer.py",
)

_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "global_scaler_fit",
        r"(?:StandardScaler|MinMaxScaler|RobustScaler)\s*\([^\n]*\)\s*\.fit(?:_transform)?\s*\([^\n]*(?:all|full|concat)",
        "疑似在全量/拼接数据上拟合 scaler，应改为每折仅训练月拟合。",
    ),
    (
        "global_zero_fill",
        r"\.fillna\s*\(\s*0(?:\.0)?\s*\)",
        "全局填零会混淆真实零与缺失，需采用折内统计量并保留缺失指示。",
    ),
    (
        "unmasked_sequence_pooling",
        r"\.mean\s*\(\s*dim\s*=\s*1\s*\)",
        "序列均值未见 mask 约束，padding 可能进入池化结果。",
    ),
    (
        "future_inclusive_event_boundary",
        r"(?:event|clock|cal|time)[^\n]*<=?[^\n]*(?:target_time|reference_time)[^\n]*\+",
        "辅助事件边界疑似包含参考时刻之后的数据，必须限制 source_time <= reference_time。",
    ),
)


def audit_teacher_script(
    workspace_root: Path, explicit_path: Path | None
) -> dict[str, Any]:
    """Statically inspect one explicitly supplied or exact-name teacher script."""

    workspace = Path(workspace_root).resolve()
    script = _resolve_script(workspace, explicit_path)
    if script is None:
        return {
            "schema_version": "data_v3.teacher_script_audit.v1",
            "status": "not_provided",
            "audited_path": None,
            "requested_file_names": list(_REQUESTED_FILE_NAMES),
            "findings": [],
            "note": "未找到老师原始训练脚本，无法核验其辅助数据时间边界。",
        }
    text = script.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []
    for name, pattern, recommendation in _RULES:
        for line_number, line in enumerate(lines, start=1):
            if re.search(pattern, line, flags=re.IGNORECASE):
                findings.append(
                    {
                        "name": name,
                        "line": line_number,
                        "evidence": line.strip()[:300],
                        "recommendation": recommendation,
                    }
                )
                break
    return {
        "schema_version": "data_v3.teacher_script_audit.v1",
        "status": "audited",
        "audited_path": str(script),
        "requested_file_names": list(_REQUESTED_FILE_NAMES),
        "findings": findings,
        "note": "静态规则审查不会执行外部训练代码；未命中的写法仍需人工复核。",
    }


def render_teacher_audit_markdown(result: dict[str, Any]) -> str:
    lines = ["# 教师脚本边界审查", "", f"状态：`{result['status']}`。", ""]
    if result["status"] == "not_provided":
        lines.extend(
            [
                result["note"],
                "",
                "待组长补充的候选文件名：" + "、".join(result["requested_file_names"]) + "。",
            ]
        )
    else:
        lines.extend([f"审查文件：`{result['audited_path']}`", ""])
        if result["findings"]:
            lines.extend(["| 检查项 | 行号 | 证据 | 建议 |", "|---|---:|---|---|"])
            for item in result["findings"]:
                evidence = str(item["evidence"]).replace("|", "\\|")
                lines.append(
                    f"| {item['name']} | {item['line']} | `{evidence}` | {item['recommendation']} |"
                )
        else:
            lines.append("未命中当前四类静态规则；仍需人工确认数据连接和时间边界。")
    return "\n".join(lines) + "\n"


def _resolve_script(workspace: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        candidate = Path(explicit_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"teacher script does not exist: {candidate}")
        return candidate
    for name in _REQUESTED_FILE_NAMES:
        matches = sorted(workspace.rglob(name), key=lambda path: str(path).lower())
        if matches:
            return matches[0].resolve()
    return None
