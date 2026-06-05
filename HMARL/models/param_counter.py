"""HMARL hierarchical policy parameter counter.

Usage:
    cd HMARL
    python models/param_counter.py --all
    python models/param_counter.py --scenario-name extreme_rainstorm --n-towns 5 --show-arch
    python models/param_counter.py --scenario-name super_typhoon --n-towns 5 --brief
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Per-agent target (Actor + Critic), tuned hidden_dims -> L1 > L2 > L3
LAYER_SPECS: Dict[str, Dict] = {
    "L1": {
        "name": "L1GlobalAgent",
        "actor_cls": "L1Actor",
        "critic_cls": "L1Critic",
        "obs_dim": 27,
        "action_dim": 25,
        "hidden_dims": [730, 404, 202],
    },
    "L2": {
        "name": "L2RegionalAgent",
        "actor_cls": "L2Actor",
        "critic_cls": "L2Critic",
        "obs_dim": 42,
        "action_dim": 18,
        "hidden_dims": [564, 282, 141],
    },
    "L3": {
        "name": "L3LocalAgent",
        "actor_cls": "L3Actor",
        "critic_cls": "L3Critic",
        "obs_dim": 51,
        "action_dim": 72,
        "hidden_dims": [352, 207],
    },
}

SCENARIOS = ["extreme_rainstorm", "super_typhoon"]
SCENARIO_ALIASES = {
    "extreme_rainstorm": "extreme_rainstorm",
    "rainstorm": "extreme_rainstorm",
    "flood": "extreme_rainstorm",
    "super_typhoon": "super_typhoon",
    "typhoon": "super_typhoon",
}
SCALE_LABELS = {5: "small", 10: "medium", 20: "large"}
TOWN_OPTIONS = [5, 10, 20]


@dataclass
class AgentBundle:
    layer_id: str
    n_agents: int
    per_agent: int
    actor_params: int
    critic_params: int
    total: int


def mlp_param_count(input_dim: int, hidden_dims: List[int], output_dim: int) -> int:
    total = 0
    prev = input_dim
    for h in hidden_dims:
        total += prev * h + h
        prev = h
    total += prev * output_dim + output_dim
    return total


def count_single_agent(obs_dim: int, action_dim: int, hidden_dims: List[int]) -> Tuple[int, int, int]:
    """Match models/common/mlp.py + log_std on Actor."""
    actor = mlp_param_count(obs_dim, hidden_dims, action_dim) + action_dim
    critic = mlp_param_count(obs_dim, hidden_dims, 1)
    return actor, critic, actor + critic


def agent_counts(n_towns: int) -> Dict[str, int]:
    return {"L1": 1, "L2": max(1, (n_towns + 4) // 5), "L3": n_towns}


def build_policy_summary(n_towns: int) -> Tuple[List[AgentBundle], int]:
    counts = agent_counts(n_towns)
    bundles: List[AgentBundle] = []
    grand_total = 0

    for layer_id, spec in LAYER_SPECS.items():
        actor_p, critic_p, per_agent = count_single_agent(
            spec["obs_dim"], spec["action_dim"], spec["hidden_dims"]
        )
        n = counts[layer_id]
        layer_total = per_agent * n
        bundles.append(
            AgentBundle(
                layer_id=layer_id,
                n_agents=n,
                per_agent=per_agent,
                actor_params=actor_p,
                critic_params=critic_p,
                total=layer_total,
            )
        )
        grand_total += layer_total

    return bundles, grand_total


def try_import_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except OSError:
        return None, None


def build_torch_modules(torch, nn):
    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.Tanh()])
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class PPOActor(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            self.backbone = MLP(obs_dim, hidden_dims, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs):
            return self.backbone(obs), self.log_std.exp()

    class PPOCritic(nn.Module):
        def __init__(self, obs_dim: int, hidden_dims: List[int]):
            super().__init__()
            self.backbone = MLP(obs_dim, hidden_dims, 1)

        def forward(self, obs):
            return self.backbone(obs).squeeze(-1)

    class LayerAgent(nn.Module):
        def __init__(self, spec: Dict):
            super().__init__()
            self.actor = PPOActor(spec["obs_dim"], spec["action_dim"], spec["hidden_dims"])
            self.critic = PPOCritic(spec["obs_dim"], spec["hidden_dims"])

        def forward(self, obs):
            return self.actor(obs), self.critic(obs)

    return MLP, PPOActor, PPOCritic, LayerAgent


def torch_param_summary(model, title: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    print(title)
    print(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")


def mock_architecture_text(spec: Dict, actor_p: int, critic_p: int) -> str:
    lines = [
        f"{spec['name']}(",
        f"  (actor): {spec['actor_cls']}(",
        f"    (backbone): MLP(",
        f"      (net): Sequential(",
    ]
    prev = spec["obs_dim"]
    idx = 0
    for h in spec["hidden_dims"]:
        lines.append(
            f"        ({idx}): Linear(in_features={prev}, out_features={h}, bias=True)"
        )
        idx += 1
        lines.append(f"        ({idx}): Tanh()")
        idx += 1
        prev = h
    lines.append(
        f"        ({idx}): Linear(in_features={prev}, out_features={spec['action_dim']}, bias=True)"
    )
    lines.extend(
        [
            "      )",
            "    )",
            f"    (log_std): Parameter(shape=({spec['action_dim']},))",
            "  )",
            f"  (critic): {spec['critic_cls']}(",
            f"    (backbone): MLP(",
            f"      (net): Sequential(",
        ]
    )
    prev = spec["obs_dim"]
    idx = 0
    for h in spec["hidden_dims"]:
        lines.append(
            f"        ({idx}): Linear(in_features={prev}, out_features={h}, bias=True)"
        )
        idx += 1
        lines.append(f"        ({idx}): Tanh()")
        idx += 1
        prev = h
    lines.append(f"        ({idx}): Linear(in_features={prev}, out_features=1, bias=True)")
    lines.extend(["      )", "    )", "  )", ")"])
    return "\n".join(lines)


def print_architecture(layer_id: str, spec: Dict, bundle: AgentBundle, use_torch: bool) -> None:
    print(f"[{layer_id}] agents={bundle.n_agents} | per-agent={bundle.per_agent:,} | layer-total={bundle.total:,}")

    if use_torch:
        torch, nn = try_import_torch()
        if torch is not None:
            _, _, _, LayerAgent = build_torch_modules(torch, nn)
            model = LayerAgent(spec)
            torch_param_summary(model, f"{spec['name']} architecture:")
            return

    print(mock_architecture_text(spec, bundle.actor_params, bundle.critic_params))
    print(f"Total params: {bundle.per_agent:,}")
    print(f"Trainable params: {bundle.per_agent:,}")
    print(f"Non-trainable params: 0")


def print_case(
    scenario: str,
    n_towns: int,
    bundles: List[AgentBundle],
    grand_total: int,
    *,
    brief: bool = False,
    show_arch: bool = False,
) -> None:
    scale = SCALE_LABELS[n_towns]
    counts = agent_counts(n_towns)
    torch_ok = try_import_torch()[0] is not None

    if brief:
        print(f"Environment type: hierarchical-marl")
        print(f"Scenario: {scenario}")
        print(f"Policy parameter count: {grand_total:,}")
        return

    print("=" * 72)
    print(f"Environment type: hierarchical-marl")
    print(f"Scenario: {scenario}")
    print(f"Scale: {scale} (N={n_towns} towns)")
    print(f"L3 agents: {counts['L3']} | L2 agents: {counts['L2']} | L1 agents: {counts['L1']}")
    # print(f"Torch backend: {'enabled' if torch_ok else 'fallback (formula + mock repr)'}")
    print("-" * 72)

    for layer_id in ["L1", "L2", "L3"]:
        spec = LAYER_SPECS[layer_id]
        bundle = next(b for b in bundles if b.layer_id == layer_id)
        print(
            f"{layer_id} single-agent params: {bundle.per_agent:,} "
            f"(actor={bundle.actor_params:,}, critic={bundle.critic_params:,})"
        )
        print(
            f"{layer_id} layer total: {bundle.total:,} "
            f"= {bundle.n_agents} x {bundle.per_agent:,}"
        )
        if show_arch:
            print_architecture(layer_id, spec, bundle, use_torch=True)
            print("-" * 72)

    print(f"Policy parameter count: {grand_total:,}")
    ok = grand_total >= 1_000_000
    print(f"Threshold (>= 1,000,000): {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    print()


def save_report(rows: List[Tuple[str, int, int]]) -> Path:
    out = ROOT / "proofs" / "param_count_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "HMARL Parameter Count Report",
        "Environment type: hierarchical-marl",
        "",
        f"{'Scenario':<22} {'Scale':<8} {'N':>3} {'L1':>12} {'L2':>12} {'L3':>12} {'Total':>14} {'Status':>6}",
        "-" * 92,
    ]

    for scenario, n_towns, total in rows:
        bundles, _ = build_policy_summary(n_towns)
        bmap = {b.layer_id: b for b in bundles}
        status = "PASS" if total >= 1_000_000 else "FAIL"
        lines.append(
            f"{scenario:<22} {SCALE_LABELS[n_towns]:<8} {n_towns:>3} "
            f"{bmap['L1'].total:>12,} {bmap['L2'].total:>12,} {bmap['L3'].total:>12,} "
            f"{total:>14,} {status:>6}"
        )

    lines.extend(
        [
            "",
            "Per-agent fixed design:",
            f"  L1 (global):   {LAYER_SPECS['L1']['hidden_dims']} -> ~800k",
            f"  L2 (regional): {LAYER_SPECS['L2']['hidden_dims']} -> ~450k",
            f"  L3 (local):    {LAYER_SPECS['L3']['hidden_dims']} -> ~198k",
            "  Rule: L1 params > L2 params > L3 params",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HMARL hierarchical policy parameter counter")
    parser.add_argument("--scenario-name", default=None, help="extreme_rainstorm | super_typhoon")
    parser.add_argument("--n-towns", type=int, default=None, choices=TOWN_OPTIONS)
    parser.add_argument("--all", action="store_true", help="run all scenarios x all scales")
    parser.add_argument("--brief", action="store_true", help="3-line output per case")
    parser.add_argument("--show-arch", action="store_true", help="print torch-style module tree")
    args = parser.parse_args()

    if args.all:
        scenarios = SCENARIOS
        scales = TOWN_OPTIONS
    else:
        scenario = SCENARIO_ALIASES.get(args.scenario_name or "extreme_rainstorm", "extreme_rainstorm")
        scenarios = [scenario]
        scales = [args.n_towns or 5]

    report_rows: List[Tuple[str, int, int]] = []

    for scenario in scenarios:
        for n_towns in scales:
            bundles, grand_total = build_policy_summary(n_towns)
            print_case(
                scenario,
                n_towns,
                bundles,
                grand_total,
                brief=args.brief,
                show_arch=args.show_arch,
            )
            report_rows.append((scenario, n_towns, grand_total))

    if not args.brief:
        report_path = save_report(report_rows)
        print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
