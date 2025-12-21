"""Shared evaluation helpers for CLI, API, and UI integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from envs import DisasterCellularEnv, MultiModalCommEnv
from models.multimodal_policy import MultimodalPolicy
from models.policy_network import MLPActorCritic


def build_env(config: Dict[str, Dict], env_type: str):
    """Instantiate either the baseline or multimodal environment."""
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def load_policy(checkpoint: Path, env, config: Dict[str, Dict], env_type: str):
    """Rebuild the policy network and load a saved checkpoint."""
    if env_type == "multimodal":
        policy = MultimodalPolicy(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=config["model"].get("multimodal_hidden_sizes", [1024, 1024, 512, 512]),
            device=config["train"].get("device", "auto"),
        )
    else:
        policy = MLPActorCritic(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=config["model"].get("hidden_sizes", [128, 128]),
            device=config["train"].get("device", "auto"),
        )
    state_dict = torch.load(checkpoint, map_location=policy.device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def apply_custom_user_state(env, user_state: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """Delegate to the environment to update user states when supported."""
    if not user_state:
        return None, None
    if hasattr(env, "apply_custom_user_state"):
        return env.apply_custom_user_state(user_state)
    raise AttributeError("Environment does not support custom user state overrides.")


def evaluate_policy(
    env,
    policy,
    episodes: int,
    deterministic: bool = True,
    render: bool = False,
    custom_user_state: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """Run rollouts and return rewards, coverages, and structured episode reports."""
    rewards: List[float] = []
    coverages: List[float] = []
    reports: List[Dict[str, Any]] = []
    scenario_meta = _describe_scenario(env)
    for episode in range(episodes):
        obs, info = env.reset()
        if custom_user_state:
            custom_obs, custom_info = apply_custom_user_state(env, custom_user_state)
            if custom_obs is not None and custom_info is not None:
                obs, info = custom_obs, custom_info
        state_snapshot = _capture_network_state(env, info)
        done = False
        ep_reward = 0.0
        final_cov = state_snapshot.get("coverage_ratio", 0.0)
        steps = 0
        last_info = info
        episode_report: Dict[str, Any] = {
            "episode": episode + 1,
            "scenario": scenario_meta,
            "initial_state": state_snapshot,
            "steps": [],
        }
        while not done:
            action, _, _ = policy.act(obs, deterministic=deterministic)
            action_value = int(action)
            prev_snapshot = state_snapshot
            obs, reward, terminated, truncated, info = env.step(action_value)
            last_info = info
            done = terminated or truncated
            ep_reward += reward
            state_snapshot = _capture_network_state(env, info)
            final_cov = float(info.get("coverage_ratio", final_cov))
            steps += 1
            episode_report["steps"].append(
                {
                    "step": steps,
                    "action_index": action_value,
                    "action_desc": _decode_multimodal_action(env, action_value),
                    "reward": float(reward),
                    "post_state": state_snapshot,
                    "coverage_delta": state_snapshot.get("coverage_ratio", 0.0)
                    - prev_snapshot.get("coverage_ratio", 0.0),
                    "broadcast_delta": state_snapshot.get("broadcast_ratio", 0.0)
                    - prev_snapshot.get("broadcast_ratio", 0.0),
                }
            )
            if render:
                print(
                    f"[Eval] Episode {episode} Step {steps} | reward={reward:+.2f} | coverage={final_cov:.2%}"
                )
                env.render()
        rewards.append(ep_reward)
        coverages.append(final_cov)
        episode_report["final_state"] = state_snapshot
        episode_report["total_reward"] = ep_reward
        episode_report["termination_reason"] = last_info.get("reason", "episode_finished")
        episode_report["steps_taken"] = steps
        reports.append(episode_report)
    return rewards, coverages, reports


def format_episode_report(report: Dict[str, Any]) -> str:
    """Return a multi-line string describing the episode."""
    scenario = report.get("scenario", {}) or {}
    header_parts = [f"Episode {report.get('episode', '?')}"]
    scenario_name = scenario.get("name")
    if scenario_name:
        header_parts.append(f"scenario={scenario_name}")
    disaster_type = scenario.get("disaster_type")
    if disaster_type:
        header_parts.append(f"type={disaster_type}")
    lines = ["\n=== " + " | ".join(header_parts) + " ==="]

    initial_state = report.get("initial_state", {})
    total_users = initial_state.get("total_users")
    connected = initial_state.get("connected_users")
    broadcast_served = initial_state.get("broadcast_served_users")
    if total_users is not None:
        connected_str = connected if connected is not None else "n/a"
        broadcast_str = broadcast_served if broadcast_served is not None else "n/a"
        lines.append(
            f"  1) Disaster network status -> connected {connected_str}/{total_users} | "
            f"broadcast-served {broadcast_str}/{total_users}"
        )
    else:
        lines.append("  1) Disaster network status -> user metrics unavailable")
    lines.append(
        "     coverage={:.2%} | broadcast={:.2%} | remaining_budget={:.1f}".format(
            initial_state.get("coverage_ratio", 0.0),
            initial_state.get("broadcast_ratio", 0.0),
            initial_state.get("remaining_budget", 0.0),
        )
    )
    lines.extend(_format_disaster_device_details(initial_state))

    steps = report.get("steps", [])
    if steps:
        lines.append("  2) Model deployment strategy:")
        for step in steps:
            action_desc = step.get("action_desc")
            if action_desc:
                location = action_desc.get("location")
                location_text = f"@{location}" if location is not None else ""
                action_text = (
                    f"site#{action_desc.get('site_index')} {location_text} | "
                    f"comm={action_desc.get('comm_mode')} | broadcast={action_desc.get('broadcast_mode')}"
                )
            else:
                action_text = f"action_index={step.get('action_index')}"
            lines.append(
                f"     Step {step.get('step'):02d}: {action_text} | reward={step.get('reward', 0.0):+.2f}"
            )
            post_state = step.get("post_state", {})
            cov_delta = step.get("coverage_delta")
            broadcast_delta = step.get("broadcast_delta")
            delta_text = ""
            if cov_delta is not None and broadcast_delta is not None:
                delta_text = f" | Δcoverage={cov_delta:+.2%} Δbroadcast={broadcast_delta:+.2%}"
            lines.append(
                "        After step -> coverage={:.2%} | broadcast={:.2%} | remaining_budget={:.1f}{}".format(
                    post_state.get("coverage_ratio", 0.0),
                    post_state.get("broadcast_ratio", 0.0),
                    post_state.get("remaining_budget", 0.0),
                    delta_text,
                )
            )
    else:
        lines.append("  2) Model deployment strategy: no actions executed.")

    final_state = report.get("final_state", initial_state)
    reason = report.get("termination_reason", "episode_finished")
    final_connected = final_state.get("connected_users")
    final_total_users = final_state.get("total_users")
    final_connected_str = final_connected if final_connected is not None else "n/a"
    final_total_users_str = final_total_users if final_total_users is not None else "n/a"
    lines.append(
        "  3) Network recovery -> coverage={:.2%} | broadcast={:.2%} | connected={}/{} | "
        "remaining_budget={:.1f} | total_reward={:.2f} | steps={} | stop_reason={}".format(
            final_state.get("coverage_ratio", 0.0),
            final_state.get("broadcast_ratio", 0.0),
            final_connected_str,
            final_total_users_str,
            final_state.get("remaining_budget", 0.0),
            report.get("total_reward", 0.0),
            report.get("steps_taken", 0),
            reason,
        )
    )
    lines.extend(_format_recovery_details(initial_state, final_state))
    return "\n".join(lines)


def _describe_scenario(env) -> Dict[str, Any]:
    scenario = getattr(env, "scenario", None)
    return {
        "name": getattr(scenario, "name", None) if scenario else None,
        "disaster_type": getattr(scenario, "disaster_type", None) if scenario else None,
        "num_users": getattr(env, "num_users", None),
        "max_steps": getattr(env, "max_steps", None),
    }


def _capture_network_state(env, info: Dict[str, float]) -> Dict[str, Any]:
    snapshot = {
        "coverage_ratio": float(info.get("coverage_ratio", 0.0)),
        "broadcast_ratio": float(info.get("broadcast_ratio", 0.0)),
        "remaining_budget": float(info.get("remaining_budget", getattr(env, "remaining_budget", 0.0))),
        "total_users": getattr(env, "num_users", None),
    }

    user_details = _extract_user_details(env)
    if user_details:
        connected = [detail.get("connected", False) for detail in user_details]
        broadcast_served = [detail.get("broadcast_served", False) for detail in user_details]
        snapshot["connected_users"] = int(sum(bool(flag) for flag in connected))
        snapshot["broadcast_served_users"] = int(sum(bool(flag) for flag in broadcast_served))
    else:
        if hasattr(env, "user_connected"):
            connected = getattr(env, "user_connected")
            snapshot["connected_users"] = int(np.count_nonzero(connected)) if connected is not None else None
        else:
            snapshot["connected_users"] = None
        if hasattr(env, "broadcast_served"):
            served = getattr(env, "broadcast_served")
            snapshot["broadcast_served_users"] = int(np.count_nonzero(served)) if served is not None else None
        else:
            snapshot["broadcast_served_users"] = None

    snapshot["user_details"] = user_details

    return snapshot


def _extract_user_details(env) -> List[Dict[str, Any]]:
    num_users = getattr(env, "num_users", None)
    if not num_users:
        return []

    positions = getattr(env, "user_positions", None)
    demands = getattr(env, "user_demands", None)
    connected = getattr(env, "user_connected", None)
    broadcast_served = getattr(env, "broadcast_served", None)

    details: List[Dict[str, Any]] = []
    for idx in range(int(num_users)):
        entry: Dict[str, Any] = {"id": idx}
        if positions is not None and len(positions) > idx:
            coords = positions[idx]
            entry["position"] = (int(coords[0]), int(coords[1]))
        if demands is not None and len(demands) > idx:
            entry["demand"] = float(demands[idx])
        if connected is not None and len(connected) > idx:
            entry["connected"] = bool(connected[idx])
        if broadcast_served is not None and len(broadcast_served) > idx:
            entry["broadcast_served"] = bool(broadcast_served[idx])
        details.append(entry)
    return details


def _decode_multimodal_action(env, action: int) -> Optional[Dict[str, Any]]:
    has_attrs = all(
        hasattr(env, attr)
        for attr in ("num_comm_modes", "num_broadcast_modes", "candidate_sites", "communication_modes", "broadcast_modes")
    )
    if not has_attrs:
        return None

    per_site_options = env.num_comm_modes * env.num_broadcast_modes
    if per_site_options <= 0:
        return None

    site_idx = action // per_site_options
    rem = action % per_site_options
    broadcast_idx = rem // env.num_comm_modes
    comm_idx = rem % env.num_comm_modes

    if site_idx >= getattr(env, "candidate_sites", 0):
        return None

    location = None
    if hasattr(env, "candidate_locations") and len(env.candidate_locations) > site_idx:
        coords = env.candidate_locations[site_idx]
        location = (int(coords[0]), int(coords[1]))

    comm_name = None
    if hasattr(env, "communication_modes") and len(env.communication_modes) > comm_idx:
        comm_name = env.communication_modes[comm_idx]

    broadcast_name = None
    if hasattr(env, "broadcast_modes") and len(env.broadcast_modes) > broadcast_idx:
        broadcast_name = env.broadcast_modes[broadcast_idx]

    return {
        "site_index": site_idx,
        "comm_index": comm_idx,
        "broadcast_index": broadcast_idx,
        "location": location,
        "comm_mode": comm_name,
        "broadcast_mode": broadcast_name,
    }


def _format_disaster_device_details(initial_state: Dict[str, Any]) -> List[str]:
    details = initial_state.get("user_details") or []
    damaged = [detail for detail in details if detail.get("connected") is False]
    if not details:
        return ["     -> Device-level data unavailable."]
    if not damaged:
        return ["     -> No disconnected devices at scenario start."]
    lines = ["     -> Damaged devices (initial):"]
    for detail in damaged:
        lines.append(f"        {_format_device_detail(detail)}")
    return lines


def _format_recovery_details(initial_state: Dict[str, Any], final_state: Dict[str, Any]) -> List[str]:
    initial_details = initial_state.get("user_details") or []
    final_details = final_state.get("user_details") or []
    if not final_details:
        return ["     -> Per-device recovery data unavailable."]
    final_by_id = {detail.get("id"): detail for detail in final_details if detail.get("id") is not None}

    restored: List[Dict[str, Any]] = []
    for detail in initial_details:
        idx = detail.get("id")
        if idx is None:
            continue
        initial_connected = bool(detail.get("connected"))
        final_detail = final_by_id.get(idx)
        if not initial_connected and final_detail and final_detail.get("connected"):
            entry = dict(final_detail)
            entry["status_note"] = "recovered"
            restored.append(entry)

    remaining_outages = [
        detail for detail in final_details if not detail.get("connected") and detail.get("id") is not None
    ]

    lines: List[str] = []
    if restored:
        lines.append("     -> Recovered devices (post-strategy):")
        for detail in restored:
            lines.append(f"        {_format_device_detail(detail)} [recovered]")
    else:
        lines.append("     -> No previously disconnected devices recovered.")

    if remaining_outages:
        lines.append("     -> Remaining outages after recovery:")
        for detail in remaining_outages:
            lines.append(f"        {_format_device_detail(detail)} [offline]")
    else:
        lines.append("     -> All devices connected after recovery.")
    return lines


def _format_device_detail(detail: Dict[str, Any]) -> str:
    idx = detail.get("id")
    pos = detail.get("position")
    demand = detail.get("demand")
    connected = detail.get("connected")
    broadcast_served = detail.get("broadcast_served")

    if isinstance(idx, (int, np.integer)):
        id_text = f"{int(idx):02d}"
    else:
        id_text = str(idx) if idx is not None else "??"
    pos_text = f"({pos[0]}, {pos[1]})" if isinstance(pos, tuple) else str(pos) if pos is not None else "n/a"
    if isinstance(demand, (int, float, np.floating)):
        demand_text = f"{float(demand):.1f} Mbps"
    else:
        demand_text = "n/a"
    conn_text = "online" if connected else "offline"
    broadcast_text = "served" if broadcast_served else "unserved"
    return f"Device#{id_text} pos={pos_text} demand={demand_text} | connected={conn_text} | broadcast={broadcast_text}"
