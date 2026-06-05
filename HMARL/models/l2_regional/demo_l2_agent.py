"""L2 多智能体冒烟测试：python -m models.l2_regional.demo_l2_agent"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.l1_global import L1Config, L1GlobalAgent
from models.l2_regional import L2Config, L2RegionalMARL


def main() -> None:
    n = 5
    l1_cfg = L1Config(n_regions=n)
    l1 = L1GlobalAgent(l1_cfg)

    l1_raw = {
        "disaster_type": 1,
        "global_inventory": np.array([10, 8, 6, 12, 4], dtype=np.float32),
        "region_severity": np.linspace(0.9, 0.2, n),
        "region_user_count": np.array([5000, 3000, 2000, 1500, 800], dtype=np.float32),
        "region_high_priority_ratio": np.full(n, 0.3, dtype=np.float32),
    }
    obs_l1 = l1.build_observation(l1_raw)
    _, _, _, l1_info = l1.act(obs_l1, l1_raw["global_inventory"])
    Q = l1_info["quota_matrix"]
    print("L1 quota matrix:\n", Q)

    l2_cfg = L2Config(n_regions=n, max_migrations=3, max_links=2)
    marl = L2RegionalMARL(l2_cfg)

    region_states = []
    for i in range(n):
        region_states.append(
            {
                "user_total": float(l1_raw["region_user_count"][i]),
                "high_priority_ratio": 0.35,
                "avg_demand_intensity": 0.6 + 0.1 * i,
                "residual_public_bw": 50.0 * (1 - l1_raw["region_severity"][i]),
                "residual_broadcast": 20.0,
                "deployed_counts": np.minimum(Q[i], Q[i] // 2),
                "severity": float(l1_raw["region_severity"][i]),
                "road_pass_rate": 0.4,
                "power_recovery_rate": 0.3,
                "l1_quota": Q[i].astype(np.float32),
            }
        )

    out = marl.act_all(region_states)
    print(f"obs_dim={l2_cfg.obs_dim}, action_dim={l2_cfg.action_dim}")
    print(f"migrations ({out['n_migrations']}):\n", out["migration_matrix"])
    print(f"links ({out['n_links']}):\n", out["link_matrix"])
    print("adjusted_quota:\n", out["adjusted_quota"])
    print("L2 params (shared):", marl.count_parameters())


if __name__ == "__main__":
    main()
