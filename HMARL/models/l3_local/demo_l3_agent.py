"""L1→L2→L3 联调冒烟：python -m models.l3_local.demo_l3_agent"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.l1_global import L1Config, L1GlobalAgent
from models.l2_regional import L2Config, L2RegionalMARL
from models.l3_local import L3Config, L3LocalAgent


def main() -> None:
    n_regions = 5
    region_id = 0

    # L1
    l1 = L1GlobalAgent(L1Config(n_regions=n_regions))
    l1_raw = {
        "disaster_type": 0,
        "global_inventory": np.array([10, 8, 6, 12, 4], dtype=np.float32),
        "region_severity": np.linspace(0.8, 0.2, n_regions),
        "region_user_count": np.full(n_regions, 2000.0),
        "region_high_priority_ratio": np.full(n_regions, 0.3),
    }
    _, _, _, l1_info = l1.act(l1.build_observation(l1_raw), l1_raw["global_inventory"])
    Q = l1_info["quota_matrix"]

    # L2
    states = [
        {
            "user_total": 2000,
            "high_priority_ratio": 0.3,
            "avg_demand_intensity": 0.6,
            "residual_public_bw": 40,
            "residual_broadcast": 15,
            "deployed_counts": Q[i] // 2,
            "severity": float(l1_raw["region_severity"][i]),
            "road_pass_rate": 0.5,
            "power_recovery_rate": 0.3,
            "l1_quota": Q[i].astype(np.float32),
        }
        for i in range(n_regions)
    ]
    l2_out = L2RegionalMARL(L2Config(n_regions=n_regions)).act_all(states)
    adjusted = l2_out["adjusted_quota"]

    # L3 单子区域
    cfg = L3Config()
    agent = L3LocalAgent(subregion_id=0, region_id=region_id, config=cfg)

    sub_raw = {
        "user_total": 400,
        "severity": 0.7,
        "terrain_complexity": 0.6,
        "avail_emergency_bs": 2,
        "avail_uav": 1,
    }
    constraints = L3LocalAgent.constraints_from_upper_layers(
        l1_quota_row=adjusted[region_id],
        l2_transfer_in=np.array([1, 0, 0, 0, 0]),
        l2_transfer_out=np.zeros(5),
        l2_link={"active": 1.0, "link_type": 0, "peer_region": 1, "deploy_grid": 3},
    )

    obs = agent.build_observation(sub_raw, constraints)
    print(f"L3 obs_dim={obs.shape[0]} (expect {cfg.obs_dim}), action_dim={cfg.action_dim}")

    action, log_prob, value, info = agent.act(
        obs,
        constraints,
        l2_links=l2_out["per_region"][region_id].get("links", []),
    )
    print(f"log_prob={log_prob:.3f}, value={value:.3f}, ok={info['constraint_ok']}")
    print("deployment sum per device:", info["decoded"]["deployment"].sum(axis=1))
    print("topology nodes:", info["topology"]["n_nodes"], "edges:", info["topology"]["n_edges"])
    print("params:", agent.count_parameters())

    out_path = ROOT / "outputs" / "topology_sub0.json"
    agent.export_topology(out_path)
    print("exported:", out_path)


if __name__ == "__main__":
    main()
