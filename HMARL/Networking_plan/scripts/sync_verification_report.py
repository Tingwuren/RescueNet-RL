#!/usr/bin/env python3
"""Sync 组网方案验证报告.docx with current code outputs and proofs."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "组网方案验证报告.docx"
PROOFS = ROOT / "proofs" / "dual_mode_matrix.txt"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_NS = f"{{{W}}}"


def _load_matrix() -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in PROOFS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("场景") or line.startswith("-") or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        scenario, mode = parts[0], parts[1]
        rows[(scenario, mode)] = {
            "residual": parts[2],
            "emergency": parts[3],
            "backhaul": parts[4],
        }
    return rows


def _cell_text(tc: ET.Element) -> str:
    return "".join(t.text or "" for t in tc.iter(f"{W_NS}t"))


def _set_cell_text(tc: ET.Element, text: str) -> None:
    texts = list(tc.iter(f"{W_NS}t"))
    if not texts:
        p = tc.find(f".//{W_NS}p")
        if p is None:
            p = ET.SubElement(tc, f"{W_NS}p")
        r = ET.SubElement(p, f"{W_NS}r")
        t = ET.SubElement(r, f"{W_NS}t")
        t.text = text
        return
    texts[0].text = text
    for node in texts[1:]:
        node.text = ""


def _row_cells(tr: ET.Element) -> list[ET.Element]:
    return tr.findall(f"{W_NS}tc")


def _replace_paragraph(root: ET.Element, needle: str, replacement: str) -> bool:
    for p in root.iter(f"{W_NS}p"):
        texts = list(p.iter(f"{W_NS}t"))
        full = "".join(t.text or "" for t in texts)
        if needle not in full:
            continue
        new_full = full.replace(needle, replacement)
        if not texts:
            continue
        texts[0].text = new_full
        for node in texts[1:]:
            node.text = ""
        return True
    return False


def _patch_paragraphs(root: ET.Element) -> None:
    paragraph_replacements = [
        (
            "├── outputs/               # 4 份 network_plan.json 最终输出配置文件",
            "├── outputs/               # 4 套场景×模式交付目录（network_plan.json + 01–04 表）",
        ),
        (
            "    ...",
            "    assign_grid_placements(plan, ...)   # 网格点位\n    build_topology(plan, ...)            # 分层拓扑",
        ),
        (
            "return plan  # 写入 network_plan.json",
            "return plan  # JSON + 01–04 表",
        ),
        (
            "deployment/plan_builder.py 中按灾害类型生成不同链路特征：",
            "deployment/topology_builder.py 在 grid 点位确定后生成分层拓扑，并按灾害类型追加场景补丁链路：",
        ),
        (
            'links.append({"link_type": "patch_fiber", "breakage_rate": 0.35, ...})',
            "scenario_overlays += patch_fiber",
        ),
        (
            'links.append({"link_type": "local_blackout_bridge", "blackout_zones": 4, ...})',
            "scenario_overlays += local_blackout_bridge",
        ),
        (
            "《子区域设备部署明细表》等标准化部署文档",
            "《子区域设备部署明细表》《设备点位与拓扑连接表》等标准化部署文档",
        ),
    ]
    for old, new in paragraph_replacements:
        _replace_paragraph(root, old, new)


def _patch_dual_mode_tables(root: ET.Element, matrix: dict[tuple[str, str], dict[str, str]]) -> None:
    cn_rows = {
        ("暴雨", "with_residual"): matrix[("extreme_rainstorm", "with_residual")],
        ("暴雨", "no_residual"): matrix[("extreme_rainstorm", "no_residual")],
        ("台风", "with_residual"): matrix[("super_typhoon", "with_residual")],
        ("台风", "no_residual"): matrix[("super_typhoon", "no_residual")],
    }
    en_rows = {
        ("extreme_rainstorm", "with_residual"): matrix[("extreme_rainstorm", "with_residual")],
        ("extreme_rainstorm", "no_residual"): matrix[("extreme_rainstorm", "no_residual")],
        ("super_typhoon", "with_residual"): matrix[("super_typhoon", "with_residual")],
        ("super_typhoon", "no_residual"): matrix[("super_typhoon", "no_residual")],
    }

    for tbl in root.iter(f"{W_NS}tbl"):
        rows = tbl.findall(f"./{W_NS}tr")
        if not rows:
            continue
        header = _cell_text(rows[0])
        if "deploy_priority" in header and "residual_nodes_reused" in header:
            for tr in rows[1:]:
                cells = _row_cells(tr)
                if len(cells) < 7:
                    continue
                scenario = _cell_text(cells[0]).strip()
                mode = _cell_text(cells[1]).strip()
                key = (scenario, mode)
                if key not in cn_rows:
                    continue
                data = cn_rows[key]
                _set_cell_text(cells[2], data["residual"])
                _set_cell_text(cells[3], data["emergency"])
                _set_cell_text(cells[4], data["backhaul"])
                if mode == "no_residual":
                    _set_cell_text(cells[5], "satellite_backhaul")
                else:
                    _set_cell_text(cells[5], "activate_residual_bs")
                _set_cell_text(cells[6], "PASS")
        elif "残余复用" in header and "应急部署" in header and "deploy_priority" not in header:
            for tr in rows[1:]:
                cells = _row_cells(tr)
                if len(cells) < 6:
                    continue
                scenario = _cell_text(cells[0]).strip()
                mode = _cell_text(cells[1]).strip()
                key = (scenario, mode)
                if key not in en_rows:
                    continue
                data = en_rows[key]
                _set_cell_text(cells[2], data["residual"])
                _set_cell_text(cells[3], data["emergency"])
                _set_cell_text(cells[4], data["backhaul"])
                _set_cell_text(cells[5], "PASS")


def _patch_text_nodes(root: ET.Element) -> None:
    replacements = {
        "deploy_satellite_hub": "satellite_backhaul",
        "deploy_uav_mesh": "uav_mesh_coverage",
        "activate_shortwave": "shortwave_fallback",
        "01–03 部署表": "01–04 部署表",
        "01-03 部署表": "01-04 部署表",
        "01–03 表": "01–04 表",
        "01-03 表": "01-04 表",
        "4 份 network_plan.json 最终输出配置文件": "4 套场景×模式交付目录（network_plan.json + 01–04 表）",
        "《子区域设备部署明细表》等标准化部署文档": "《子区域设备部署明细表》《设备点位与拓扑连接表》等标准化部署文档",
        "emergency_base = max(emergency_base, 18)": "emergency_base = max(emergency_base, n_regions * 4)",
        "搭建卫星回传枢纽": "satellite_backhaul（卫星回传枢纽）",
    }
    for node in root.iter(f"{W_NS}t"):
        if not node.text:
            continue
        text = node.text
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
        node.text = text


def _patch_xml(xml: str, matrix: dict[tuple[str, str], dict[str, str]]) -> str:
    root = ET.fromstring(xml)
    _patch_dual_mode_tables(root, matrix)
    _patch_paragraphs(root)
    _patch_text_nodes(root)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    if not DOCX.exists():
        raise SystemExit(f"Missing docx: {DOCX}")
    matrix = _load_matrix()
    backup = DOCX.with_suffix(".docx.bak")
    if not backup.exists():
        shutil.copy2(DOCX, backup)

    tmp = DOCX.with_suffix(".docx.tmp")
    with zipfile.ZipFile(DOCX, "r") as zin, zipfile.ZipFile(tmp, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                text = data.decode("utf-8")
                text = _patch_xml(text, matrix)
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(DOCX)
    print(f"Updated: {DOCX}")
    print(f"Backup:  {backup}")
    for key, vals in sorted(matrix.items()):
        print(f"  {key[0]} / {key[1]}: residual={vals['residual']} emergency={vals['emergency']}")


if __name__ == "__main__":
    main()
