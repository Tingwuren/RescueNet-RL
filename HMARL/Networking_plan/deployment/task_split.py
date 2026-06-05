"""Task split: region -> node -> business (design doc 8.2)."""

from __future__ import annotations

from typing import Any, Dict, List


def split_tasks(
    n_regions: int,
    nodes: List[Dict[str, Any]],
    business_tasks: List[str],
) -> List[Dict[str, Any]]:
    """Assign business tasks to regional node groups."""
    tasks: List[Dict[str, Any]] = []
    for region_id in range(n_regions):
        region_nodes = [n for n in nodes if n.get("region_id") == region_id]
        tasks.append(
            {
                "region_id": region_id,
                "node_count": len(region_nodes),
                "business_tasks": business_tasks,
                "nodes": [n["id"] for n in region_nodes],
            }
        )
    return tasks
