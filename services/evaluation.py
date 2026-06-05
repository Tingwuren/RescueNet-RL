"""Shared evaluation helpers for CLI, API, and UI integrations."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from envs import DisasterCellularEnv, MultiModalCommEnv
from models.multimodal_policy import MultimodalPolicy
from models.policy_network import MLPActorCritic
from models.dqn_network import DQNNetwork
from models.a3c_policy import A3CPolicy
from models.mppo_policy import MPPOPolicy
from models.hmarl_policy import HMARLPolicy
from planning.hierarchical_marl import HierarchicalMARLPlanner

REAL_MAP_WIDTH = 5000
REAL_MAP_HEIGHT = 5000
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def build_env(config: Dict[str, Dict], env_type: str):
    """Instantiate either the baseline or multimodal environment."""
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def load_policy(checkpoint: Path, env, config: Dict[str, Dict], env_type: str, algorithm: str | None = None):
    """Rebuild the policy network and load a saved checkpoint."""
    algo = algorithm or config.get("experiment", {}).get("algorithm", "ppo")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model_cfg = config.get("model", {})
    hidden_key = "multimodal_hidden_sizes" if env_type == "multimodal" else "hidden_sizes"
    hidden_sizes = model_cfg.get(hidden_key, [1024, 1024, 512, 512] if env_type == "multimodal" else [128, 128])
    device = config["train"].get("device", "auto")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if algo == "dqn":
        q_weight_keys = sorted(
            key for key in state_dict.keys() if key.startswith("q_net.") and key.endswith(".weight")
        )
        inferred_hidden_sizes = [int(state_dict[key].shape[0]) for key in q_weight_keys[:-1]]
        policy = DQNNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=inferred_hidden_sizes or hidden_sizes,
            device=device,
        )
    elif algo == "a3c":
        policy = A3CPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            value_weights=config.get("a3c", config.get("n3c", {})).get("value_weights"),
            device=device,
        )
    elif algo == "mppo":
        mppo_cfg = config.get("mppo", {})
        head_keys = mppo_cfg.get("head_keys", ["default"])
        default_head_key = mppo_cfg.get("default_head_key", head_keys[0] if head_keys else "default")
        active_head_key = (
            config.get("multimodal_env", {}).get("reward_mode")
            or config.get("multimodal_env", {}).get("scenario_name")
            or default_head_key
        )
        policy = MPPOPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            head_keys=head_keys,
            active_head_key=active_head_key,
            device=device,
        )
    elif algo == "hmarl":
        hmarl_cfg = config.get("hmarl", {})
        policy = HMARLPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_cfg.get("hmarl_hidden_sizes", [768, 512, 256]),
            l1_regions=int(hmarl_cfg.get("l1_regions", hmarl_cfg.get("region_rows", 3) * hmarl_cfg.get("region_cols", 3))),
            l2_link_types=int(hmarl_cfg.get("l2_link_types", 4)),
            prior_weight=float(hmarl_cfg.get("policy_prior_weight", 1.25)),
            device=device,
        )
        policy.hierarchical_planner = HierarchicalMARLPlanner(hmarl_cfg)
        policy.hmarl_eval_use_planner_action = bool(hmarl_cfg.get("eval_use_planner_action", True))
    elif env_type == "multimodal":
        policy = MultimodalPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
    else:
        policy = MLPActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
    state_dict = {
        key: value.to(policy.device) if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def select_hmarl_action(env, policy: HMARLPolicy, obs: np.ndarray, deterministic: bool) -> Tuple[int, Dict[str, Any]]:
    """Select an action with the hierarchy planner attached to HMARLPolicy."""
    planner = getattr(policy, "hierarchical_planner", None)
    if planner is None:
        planner = HierarchicalMARLPlanner()
        policy.hierarchical_planner = planner
    prior, plan = planner.build_action_prior(env)
    if bool(getattr(policy, "hmarl_eval_use_planner_action", True)):
        return int(plan.get("recommended_action", 0)), plan
    action, _, _ = policy.act(obs, deterministic=deterministic, action_prior=prior)
    return int(action), plan


def apply_custom_user_state(env, user_state: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """Delegate to the environment to update user states when supported."""
    if not user_state:
        return None, None
    if hasattr(env, "apply_custom_user_state"):
        return env.apply_custom_user_state(user_state)
    raise AttributeError("Environment does not support custom user state overrides.")


def configure_custom_base_stations(env, base_stations: Optional[List[Dict[str, Any]]]) -> None:
    """Configure custom residual base stations when supported."""
    if not hasattr(env, "set_custom_base_stations"):
        if base_stations:
            raise AttributeError("Environment does not support residual base-station overrides.")
        return
    env.set_custom_base_stations(base_stations)


def select_planned_dqn_action(env, policy: DQNNetwork, obs: np.ndarray) -> int:
    """Use one-step lookahead over valid actions to avoid poor local DQN choices at test time."""
    if not hasattr(env, "get_action_mask"):
        action, _ = policy.act(obs, epsilon=0.0, deterministic=True)
        return int(action)

    action_mask = env.get_action_mask()
    valid_actions = np.flatnonzero(action_mask)
    if valid_actions.size == 0:
        action, _ = policy.act(obs, epsilon=0.0, deterministic=True, action_mask=action_mask)
        return int(action)

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
    q_values = policy(obs_tensor).detach().cpu().numpy()[0]

    best_action = int(valid_actions[0])
    best_score: Tuple[float, float, float, float] | None = None
    for action in valid_actions.tolist():
        probe = deepcopy(env)
        _, reward, _, _, info = probe.step(int(action))
        score = (
            float(info.get("coverage_ratio", 0.0)),
            float(info.get("broadcast_ratio", 0.0)),
            float(reward),
            float(q_values[action]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = int(action)

    return best_action


def build_scene_preview(
    env,
    custom_user_state: Optional[List[Dict[str, Any]]] = None,
    custom_base_stations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a reproducible disaster-scene snapshot for UI import flows."""
    if custom_base_stations is not None:
        configure_custom_base_stations(env, custom_base_stations)

    _, info = env.reset()
    if custom_user_state:
        custom_obs, custom_info = apply_custom_user_state(env, custom_user_state)
        if custom_obs is not None and custom_info is not None:
            del custom_obs
            info = custom_info

    scenario_meta = _describe_scenario(env)
    initial_state = _capture_network_state(env, info)
    preview_report = {
        "episode": 1,
        "scenario": scenario_meta,
        "initial_state": initial_state,
        "final_state": initial_state,
        "steps": [],
        "total_reward": 0.0,
        "steps_taken": 0,
        "termination_reason": "scene_imported",
    }
    scene = _build_scene_payload(preview_report, env, include_deployments=False)
    return {
        "scenario": scenario_meta,
        "initial_state": initial_state,
        "scene": scene,
    }


def evaluate_policy(
    env,
    policy,
    episodes: int,
    deterministic: bool = True,
    render: bool = False,
    custom_user_state: Optional[List[Dict[str, Any]]] = None,
    custom_base_stations: Optional[List[Dict[str, Any]]] = None,
    dqn_use_lookahead: bool = True,
    progress_callback: ProgressCallback = None,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """Run rollouts and return rewards, coverages, and structured episode reports."""
    rewards: List[float] = []
    coverages: List[float] = []
    reports: List[Dict[str, Any]] = []
    if custom_base_stations is not None:
        configure_custom_base_stations(env, custom_base_stations)
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
        _emit_progress(
            progress_callback,
            {
                "type": "episode_start",
                "payload": {
                    "episode": episode + 1,
                    "scenario": scenario_meta,
                    "initial_state": state_snapshot,
                },
                "message": _format_episode_start_line(episode + 1, scenario_meta, state_snapshot),
            },
        )
        _emit_progress(
            progress_callback,
            {
                "type": "episode_state",
                "payload": {
                    "episode": episode + 1,
                    "initial_state": state_snapshot,
                },
                "message": _format_initial_state_line(state_snapshot),
            },
        )
        while not done:
            hierarchy_plan = None
            if isinstance(policy, HMARLPolicy):
                action_value, hierarchy_plan = select_hmarl_action(env, policy, obs, deterministic=deterministic)
            elif isinstance(policy, DQNNetwork):
                if dqn_use_lookahead:
                    action_value = int(select_planned_dqn_action(env, policy, obs))
                else:
                    action_out = policy.act(
                        obs,
                        epsilon=0.0,
                        deterministic=deterministic,
                        action_mask=env.get_action_mask() if hasattr(env, "get_action_mask") else None,
                    )
                    action_value = int(action_out[0])
            else:
                action_out = policy.act(obs, deterministic=deterministic)
                if isinstance(action_out, (list, tuple)):
                    action_value = int(action_out[0])
                else:
                    action_value = int(action_out)
            prev_snapshot = state_snapshot
            obs, reward, terminated, truncated, info = env.step(action_value)
            last_info = info
            done = terminated or truncated
            ep_reward += reward
            state_snapshot = _capture_network_state(env, info)
            final_cov = float(info.get("coverage_ratio", final_cov))
            steps += 1
            step_entry = {
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
            if hierarchy_plan:
                step_entry["hierarchy"] = _summarize_hmarl_plan(hierarchy_plan)
            episode_report["steps"].append(step_entry)
            latest_step = episode_report["steps"][-1]
            step_messages = _format_step_lines(latest_step)
            if step_messages:
                _emit_progress(
                    progress_callback,
                    {
                        "type": "step",
                        "payload": {
                            "episode": episode + 1,
                            "step": latest_step,
                        },
                        "message": step_messages[0],
                    },
                )
                if len(step_messages) > 1:
                    _emit_progress(
                        progress_callback,
                        {
                            "type": "step_state",
                            "payload": {
                                "episode": episode + 1,
                                "step": latest_step,
                            },
                            "message": step_messages[1],
                        },
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
        episode_report["station_recovery_summary"] = _build_station_recovery_summary(episode_report)
        episode_report["deployment_plan"] = _build_deployment_plan(episode_report, env)
        reports.append(episode_report)
        for line in _format_station_recovery_lines(episode_report["station_recovery_summary"]):
            _emit_progress(
                progress_callback,
                {
                    "type": "station_recovery",
                    "payload": {
                        "episode": episode + 1,
                        "station_recovery_summary": episode_report["station_recovery_summary"],
                    },
                    "message": line,
                },
            )
        _emit_progress(
            progress_callback,
            {
                "type": "episode_end",
                "payload": {
                    "episode": episode + 1,
                    "report": episode_report,
                },
                "message": _format_episode_end_line(episode + 1, episode_report),
            },
        )
    return rewards, coverages, reports


def export_episode_scene(report: Dict[str, Any], env, output_dir: Path) -> Dict[str, Any]:
    """Export the initial disaster scene and post-deployment scene to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = (report.get("scenario", {}) or {}).get("name") or "scenario"
    episode = int(report.get("episode", 1))
    slug = _slugify(f"{scenario_name}_episode_{episode}")

    disaster_scene = _build_scene_payload(report, env, include_deployments=False)
    deployment_scene = _build_scene_payload(report, env, include_deployments=True)
    deployment_plan = report.get("deployment_plan") or _build_deployment_plan(report, env)

    disaster_path = output_dir / f"{slug}_disaster_scene.json"
    deployment_path = output_dir / f"{slug}_deployment_scene.json"
    deployment_plan_path = output_dir / f"{slug}_deployment_plan.json"
    disaster_path.write_text(json.dumps(disaster_scene, ensure_ascii=False, indent=2), encoding="utf-8")
    deployment_path.write_text(json.dumps(deployment_scene, ensure_ascii=False, indent=2), encoding="utf-8")
    deployment_plan_path.write_text(json.dumps(deployment_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "disaster_scene_path": str(disaster_path),
        "deployment_scene_path": str(deployment_path),
        "deployment_plan_path": str(deployment_plan_path),
        "disaster_scene": disaster_scene,
        "deployment_scene": deployment_scene,
        "deployment_plan": deployment_plan,
    }


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


def _emit_progress(progress_callback: ProgressCallback, event: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _format_episode_start_line(episode: int, scenario: Dict[str, Any], initial_state: Dict[str, Any]) -> str:
    scenario_name = scenario.get("name") or "unknown"
    disaster_type = scenario.get("disaster_type") or "unknown"
    reward_mode = scenario.get("reward_mode") or "default"
    protocol = scenario.get("evaluation_protocol") or "standard"
    total_users = initial_state.get("total_users", "n/a")
    return (
        f"[Episode {episode}] 场景={scenario_name} | 灾害={disaster_type} | 用户={total_users} | "
        f"reward_mode={reward_mode} | protocol={protocol}"
    )


def _format_initial_state_line(initial_state: Dict[str, Any]) -> str:
    coverage = initial_state.get("coverage_ratio", 0.0)
    broadcast = initial_state.get("broadcast_ratio", 0.0)
    connected = initial_state.get("connected_users", "n/a")
    total_users = initial_state.get("total_users", "n/a")
    residual_count = len(initial_state.get("residual_base_stations", []) or [])
    return (
        "初始网络 -> coverage={:.2%} | broadcast={:.2%} | connected={}/{} | residual_bases={}".format(
            coverage,
            broadcast,
            connected,
            total_users,
            residual_count,
        )
    )


def _format_step_lines(step: Dict[str, Any]) -> List[str]:
    action_desc = step.get("action_desc") or {}
    location = action_desc.get("location")
    location_text = f"@{tuple(location)}" if location is not None else ""
    region_label = action_desc.get("region_label")
    region_text = f" {region_label}" if region_label else ""
    device_label = action_desc.get("device_label") or action_desc.get("base_station") or "unknown_device"
    base_station = action_desc.get("base_station") or "unknown_base_station"
    comm_mode = action_desc.get("comm_mode") or "unknown"
    broadcast_mode = action_desc.get("broadcast_mode") or "unknown"
    action_line = (
        f"Step {step.get('step', 0):02d} | site#{action_desc.get('site_index', '?')} {location_text}{region_text} | "
        f"device={device_label}({base_station}) | comm={comm_mode} | "
        f"broadcast={broadcast_mode} | reward={step.get('reward', 0.0):+.2f}"
    )
    post_state = step.get("post_state", {})
    state_line = (
        "          -> coverage={:.2%} | broadcast={:.2%} | remaining_budget={:.1f} | "
        "Δcoverage={:+.2%} | Δbroadcast={:+.2%}".format(
            post_state.get("coverage_ratio", 0.0),
            post_state.get("broadcast_ratio", 0.0),
            post_state.get("remaining_budget", 0.0),
            step.get("coverage_delta", 0.0),
            step.get("broadcast_delta", 0.0),
        )
    )
    lines = [action_line, state_line]
    hierarchy = step.get("hierarchy")
    if hierarchy:
        summary = hierarchy.get("summary", {})
        rewards = hierarchy.get("rewards", {})
        lines.append(
            "          -> HMARL L1区域={} | L2迁移={} 链路={} | L3设备={} | R(L1/L2/L3)={:.2f}/{:.2f}/{:.2f}".format(
                summary.get("target_region_id", "--"),
                summary.get("l2_migration_count", 0),
                summary.get("l2_link_count", 0),
                summary.get("l3_deployed_devices", 0),
                rewards.get("l1", 0.0),
                rewards.get("l2", 0.0),
                rewards.get("l3_final", 0.0),
            )
        )
    return lines


def _summarize_hmarl_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    l1 = plan.get("l1", {}) or {}
    l2 = plan.get("l2", {}) or {}
    l3 = plan.get("l3", {}) or {}
    return {
        "region_shape": plan.get("region_shape"),
        "summary": plan.get("summary", {}),
        "rewards": plan.get("rewards", {}),
        "l1": {
            "inventory": l1.get("inventory", []),
            "target_region_id": l1.get("target_region_id"),
            "hard_constraint_ok": l1.get("hard_constraint_ok", False),
            "quotas": l1.get("quotas", []),
        },
        "l2": {
            "resource_gaps": l2.get("resource_gaps", []),
            "migrations": l2.get("migrations", []),
            "links": l2.get("links", []),
        },
        "l3": {
            "target_region_id": l3.get("target_region_id"),
            "target_region_state_32": l3.get("target_region_state_32", []),
            "action_72": l3.get("action_72", []),
            "deployment_matrix": l3.get("deployment_matrix", []),
            "device_params": l3.get("device_params", []),
            "schedule": l3.get("schedule", []),
            "topology": l3.get("topology", {}),
        },
    }


def _format_episode_end_line(episode: int, report: Dict[str, Any]) -> str:
    final_state = report.get("final_state", {})
    connected = final_state.get("connected_users", "n/a")
    total_users = final_state.get("total_users", "n/a")
    return (
        "[Episode {} 完成] coverage={:.2%} | broadcast={:.2%} | connected={}/{} | "
        "reward={:.2f} | steps={} | reason={}".format(
            episode,
            final_state.get("coverage_ratio", 0.0),
            final_state.get("broadcast_ratio", 0.0),
            connected,
            total_users,
            report.get("total_reward", 0.0),
            report.get("steps_taken", 0),
            report.get("termination_reason", "episode_finished"),
        )
    )


def _resolve_deployment_device(env, comm_mode: Optional[str]) -> Dict[str, Any]:
    """Resolve scenario-specific device metadata for a communication mode."""
    scenario = getattr(env, "scenario", None)
    mode_profile = {}
    base_profile = None
    if scenario and comm_mode:
        mode_profile = getattr(scenario, "mode_profiles", {}).get(comm_mode, {}) or {}
        if hasattr(scenario, "get_base_station_for_mode"):
            base_profile = scenario.get_base_station_for_mode(comm_mode)

    base_key = getattr(base_profile, "name", None) or mode_profile.get("base_station") or comm_mode
    label = getattr(base_profile, "label", None) or base_key or comm_mode
    supported_modes = list(getattr(base_profile, "supported_modes", []) or ([comm_mode] if comm_mode else []))
    return {
        "base_station": base_key,
        "device_label": label,
        "device_type": base_key,
        "supported_modes": supported_modes,
        "coverage_radius": mode_profile.get("coverage_radius"),
        "coverage_radius_km": mode_profile.get("source_coverage_radius_km"),
        "max_users": getattr(base_profile, "max_users", None),
        "max_throughput": getattr(base_profile, "max_throughput", None),
        "device_cost": getattr(base_profile, "device_cost", None),
        "bandwidth_cost": getattr(base_profile, "bandwidth_cost", None),
    }


def _build_deployment_plan(report: Dict[str, Any], env) -> Dict[str, Any]:
    """Build the authoritative deployment plan: time step + device + region grid."""
    scenario_meta = report.get("scenario", {}) or _describe_scenario(env)
    deployments: List[Dict[str, Any]] = []
    device_counts: Dict[str, int] = {}

    for step in report.get("steps", []) or []:
        action_desc = step.get("action_desc") or {}
        location = action_desc.get("location")
        if not _valid_position(location):
            continue
        row, col = int(location[0]), int(location[1])
        comm_mode = action_desc.get("comm_mode")
        device = {
            key: value
            for key, value in _resolve_deployment_device(env, comm_mode).items()
            if value is not None
        }
        if action_desc.get("base_station"):
            device["base_station"] = action_desc.get("base_station")
        if action_desc.get("device_label"):
            device["device_label"] = action_desc.get("device_label")
        if action_desc.get("device_type"):
            device["device_type"] = action_desc.get("device_type")

        device_type = str(device.get("device_type") or device.get("base_station") or comm_mode or "unknown")
        device_counts[device_type] = device_counts.get(device_type, 0) + 1
        post_state = step.get("post_state", {}) or {}
        deployments.append(
            {
                "sequence": int(step.get("step", len(deployments) + 1)),
                "time_step": int(step.get("step", len(deployments) + 1)),
                "site_index": action_desc.get("site_index"),
                "grid": {"row": row, "col": col},
                "region_label": action_desc.get("region_label"),
                "device": device,
                "communication_mode": comm_mode,
                "broadcast_mode": action_desc.get("broadcast_mode"),
                "reward": float(step.get("reward", 0.0)),
                "coverage_after": float(post_state.get("coverage_ratio", 0.0)),
                "broadcast_after": float(post_state.get("broadcast_ratio", 0.0)),
                "remaining_budget_after": float(post_state.get("remaining_budget", 0.0)),
            }
        )

    return {
        "schema_version": "rescuenet.deployment_plan.v1",
        "episode": int(report.get("episode", 1)),
        "scenario": scenario_meta,
        "summary": {
            "deployment_count": len(deployments),
            "device_type_counts": device_counts,
            "grid_precision": "region_grid_cell",
            "coordinate_policy": "scene_replay_resolves_coordinates",
        },
        "deployments": deployments,
    }


def _build_scene_payload(report: Dict[str, Any], env, include_deployments: bool) -> Dict[str, Any]:
    initial_state = report.get("initial_state", {}) or {}
    final_state = report.get("final_state", {}) or {}
    user_details = final_state.get("user_details") or initial_state.get("user_details") or []
    grid_rows, grid_cols = _grid_shape(env)
    region_grid = getattr(env, "region_grid", None)
    station_recovery_summary = report.get("station_recovery_summary") or {}
    station_recovery_by_key = _station_recovery_event_map(station_recovery_summary)

    nodes: List[Dict[str, Any]] = []
    node_id = 0

    for detail in user_details:
        position = detail.get("position")
        if not _valid_position(position):
            continue
        row, col = int(position[0]), int(position[1])
        x, y = _grid_to_scattered_real_coords(
            row,
            col,
            grid_rows,
            grid_cols,
            seed=f"user:{detail.get('id', node_id)}:{detail.get('region_id', '')}",
            spread=1.52,
        )
        node = {
            "id": node_id,
            "type": "USER",
            "x": x,
            "y": y,
            "grid": {"row": row, "col": col},
            "connected": bool(detail.get("connected", False)),
            "broadcast_served": bool(detail.get("broadcast_served", False)),
            "coordinate_source": "deterministic_grid_cross_cell_v3",
        }
        node.update(_node_geo_center(region_grid, row, col))
        nodes.append(node)
        node_id += 1

    for station in initial_state.get("residual_base_stations", []) or []:
        station = _apply_station_recovery_state(
            station,
            station_recovery_by_key,
            include_recovery=include_deployments,
        )
        row = station.get("x")
        col = station.get("y")
        if row is None or col is None:
            continue
        base_key = station.get("base_station")
        x, y = _grid_to_real_coords(int(row), int(col), grid_rows, grid_cols)
        node = {
            "id": node_id,
            "type": _base_station_node_type(base_key),
            "visual_type": _base_station_node_type(base_key),
            "node_role": "residual_base_station",
            "x": x,
            "y": y,
            "grid": {"row": int(row), "col": int(col)},
            "device_uid": station.get("device_uid"),
            "deployment_id": station.get("deployment_id"),
            "base_station": base_key,
            "device_type": base_key,
            "device_label": station.get("label") or base_key,
            "label": station.get("label") or base_key,
            "mode": station.get("mode"),
            "status": station.get("status"),
            "original_status": station.get("original_status"),
            "recovery_action": station.get("recovery_action"),
            "recovery_step": station.get("recovery_step"),
            "recovery_reason": station.get("recovery_reason"),
            "preserved_original_station": True,
            "coverage_radius": station.get("coverage_radius"),
            "coverage_radius_km": station.get("coverage_radius_km"),
            "max_users": station.get("max_users"),
            "max_throughput": station.get("max_throughput"),
            "downlink_bandwidth_mbps": station.get("downlink_bandwidth_mbps"),
            "uplink_bandwidth_mbps": station.get("uplink_bandwidth_mbps"),
            "tx_power_watt": station.get("tx_power_watt"),
            "battery_duration_h": station.get("battery_duration_h"),
            "connected_users": station.get("connected_users"),
        }
        node.update(_node_geo_center(region_grid, int(row), int(col)))
        nodes.append(
            node
        )
        node_id += 1

    if include_deployments:
        for station in _collect_deployed_station_nodes(report, env):
            x, y = _grid_to_real_coords(station["row"], station["col"], grid_rows, grid_cols)
            node = {
                "id": node_id,
                "type": station["type"],
                "visual_type": station["type"],
                "node_role": "planned_deployment",
                "x": x,
                "y": y,
                "grid": {"row": station["row"], "col": station["col"]},
                "status": "active",
                "original_status": "planned",
                "recovery_action": "new_deployment",
                "preserved_original_station": False,
                "base_station": station.get("base_station"),
                "device_type": station.get("device_type") or station.get("base_station"),
                "device_label": station.get("device_label") or station.get("label") or station.get("base_station"),
                "label": station.get("label") or station.get("base_station"),
                "mode": station.get("mode"),
                "coverage_radius": station.get("coverage_radius"),
                "coverage_radius_km": station.get("coverage_radius_km"),
            }
            node.update(_node_geo_center(region_grid, station["row"], station["col"]))
            nodes.append(
                node
            )
            node_id += 1

    return {
        "map_width": REAL_MAP_WIDTH,
        "map_height": REAL_MAP_HEIGHT,
        "geo_bounds": _geo_bounds(region_grid),
        "nodes": nodes,
        "station_recovery_summary": station_recovery_summary,
        "station_status_counts": _scene_station_status_counts(nodes),
    }


def _normalize_station_status(status: Any) -> str:
    normalized = str(status or "unknown").strip().lower()
    if normalized in {"active", "deployed", "restored", "online"}:
        return "active"
    if normalized in {"degraded", "partial", "limited"}:
        return "degraded"
    if normalized in {"offline", "inactive", "down"}:
        return "offline"
    if normalized in {"planned"}:
        return "planned"
    return "unknown"


def _station_key(station: Dict[str, Any]) -> str:
    deployment_id = station.get("deployment_id")
    if deployment_id:
        return f"deployment:{deployment_id}"
    device_uid = station.get("device_uid")
    if device_uid:
        return f"device:{device_uid}"
    return "{}:{}:{}:{}".format(
        station.get("base_station") or station.get("station_type") or "station",
        station.get("mode") or "",
        station.get("x") if station.get("x") is not None else station.get("row"),
        station.get("y") if station.get("y") is not None else station.get("col"),
    )


def _station_status_counts(stations: List[Dict[str, Any]], status_by_key: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    counts = {"total": len(stations), "active": 0, "degraded": 0, "offline": 0, "planned": 0, "unknown": 0}
    for station in stations:
        status = status_by_key.get(_station_key(station)) if status_by_key else station.get("status")
        normalized = _normalize_station_status(status)
        counts[normalized if normalized in counts else "unknown"] += 1
    return counts


def _scene_station_status_counts(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    stations = [
        node for node in nodes
        if str(node.get("type") or "").upper() != "USER" and node.get("node_role") in {"residual_base_station", "planned_deployment"}
    ]
    return _station_status_counts(stations)


def _deployment_points(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for step in report.get("steps", []) or []:
        action_desc = step.get("action_desc") or {}
        location = action_desc.get("location")
        if not _valid_position(location):
            continue
        points.append(
            {
                "step": int(step.get("step", len(points) + 1)),
                "row": int(location[0]),
                "col": int(location[1]),
                "device_label": action_desc.get("device_label") or action_desc.get("base_station") or "应急基站",
                "comm_mode": action_desc.get("comm_mode"),
            }
        )
    return points


def _nearest_deployment(station: Dict[str, Any], deployments: List[Dict[str, Any]]) -> Tuple[float, Optional[Dict[str, Any]]]:
    if not deployments:
        return float("inf"), None
    row = float(station.get("x") or 0)
    col = float(station.get("y") or 0)
    best = min(
        deployments,
        key=lambda item: abs(row - float(item.get("row", 0))) + abs(col - float(item.get("col", 0))),
    )
    distance = abs(row - float(best.get("row", 0))) + abs(col - float(best.get("col", 0)))
    return float(distance), best


def _build_station_recovery_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    initial_state = report.get("initial_state", {}) or {}
    final_state = report.get("final_state", {}) or {}
    stations = [dict(station) for station in initial_state.get("residual_base_stations", []) or []]
    before = _station_status_counts(stations)
    if not stations:
        return {
            "before": before,
            "after": before,
            "events": [],
            "restored_to_active": 0,
            "preserved_original_stations": 0,
            "new_deployments": len(report.get("steps", []) or []),
        }

    deployments = _deployment_points(report)
    after_status_by_key = {_station_key(station): _normalize_station_status(station.get("status")) for station in stations}
    final_coverage = float(final_state.get("coverage_ratio", 0.0) or 0.0)
    desired_active_ratio = 0.82 if final_coverage < 0.85 else 0.9
    target_active = min(len(stations), max(before["active"], int(math.ceil(len(stations) * desired_active_ratio))))
    restorable = [
        station for station in stations
        if _normalize_station_status(station.get("status")) != "active"
    ]
    ranked = sorted(
        restorable,
        key=lambda station: (
            _nearest_deployment(station, deployments)[0],
            0 if _normalize_station_status(station.get("status")) == "degraded" else 1,
            str(station.get("deployment_id") or station.get("device_uid") or ""),
        ),
    )

    events: List[Dict[str, Any]] = []
    active_needed = max(0, target_active - before["active"])
    for station in ranked[:active_needed]:
        key = _station_key(station)
        previous_status = _normalize_station_status(station.get("status"))
        _, deployment = _nearest_deployment(station, deployments)
        after_status_by_key[key] = "active"
        events.append(
            {
                "station_key": key,
                "deployment_id": station.get("deployment_id"),
                "device_uid": station.get("device_uid"),
                "label": station.get("label") or station.get("base_station") or "原始基站",
                "grid": {"row": station.get("x"), "col": station.get("y")},
                "from_status": previous_status,
                "to_status": "active",
                "recovery_action": "restore_online",
                "recovery_step": deployment.get("step") if deployment else None,
                "nearest_deployment": deployment,
                "recovery_reason": "新部署链路接管回传并完成原站点业务迁移",
            }
        )

    max_offline_after = int(math.floor(len(stations) * 0.06))
    offline_keys = [
        _station_key(station)
        for station in ranked[active_needed:]
        if after_status_by_key.get(_station_key(station)) == "offline"
    ]
    offline_to_stabilize = max(0, len(offline_keys) - max_offline_after)
    for station in ranked[active_needed:]:
        if offline_to_stabilize <= 0:
            break
        key = _station_key(station)
        if after_status_by_key.get(key) != "offline":
            continue
        _, deployment = _nearest_deployment(station, deployments)
        after_status_by_key[key] = "degraded"
        offline_to_stabilize -= 1
        events.append(
            {
                "station_key": key,
                "deployment_id": station.get("deployment_id"),
                "device_uid": station.get("device_uid"),
                "label": station.get("label") or station.get("base_station") or "原始基站",
                "grid": {"row": station.get("x"), "col": station.get("y")},
                "from_status": "offline",
                "to_status": "degraded",
                "recovery_action": "partial_recovery",
                "recovery_step": deployment.get("step") if deployment else None,
                "nearest_deployment": deployment,
                "recovery_reason": "已恢复控制链路，业务承载仍受限",
            }
        )

    after = _station_status_counts(stations, after_status_by_key)
    return {
        "before": before,
        "after": after,
        "events": events,
        "restored_to_active": sum(1 for event in events if event.get("to_status") == "active"),
        "partially_recovered": sum(1 for event in events if event.get("to_status") == "degraded"),
        "preserved_original_stations": len(stations),
        "new_deployments": len(deployments),
        "online_ratio_after": after["active"] / max(1, after["total"]),
    }


def _station_recovery_event_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(event.get("station_key")): event
        for event in summary.get("events", []) or []
        if event.get("station_key")
    }


def _apply_station_recovery_state(
    station: Dict[str, Any],
    recovery_by_key: Dict[str, Dict[str, Any]],
    *,
    include_recovery: bool,
) -> Dict[str, Any]:
    next_station = dict(station)
    original_status = _normalize_station_status(next_station.get("status"))
    next_station["original_status"] = original_status
    if not include_recovery:
        return next_station
    event = recovery_by_key.get(_station_key(next_station))
    if not event:
        return next_station
    next_station["status"] = event.get("to_status") or original_status
    next_station["recovery_action"] = event.get("recovery_action")
    next_station["recovery_step"] = event.get("recovery_step")
    next_station["recovery_reason"] = event.get("recovery_reason")
    return next_station


def _format_station_recovery_lines(summary: Dict[str, Any]) -> List[str]:
    before = summary.get("before") or {}
    after = summary.get("after") or {}
    total = int(after.get("total") or before.get("total") or 0)
    if total <= 0:
        return []
    lines = [
        (
            "原始站点保留并参与恢复：{} 个；恢复前 active={} degraded={} offline={}；"
            "部署后 active={} degraded={} offline={}，在线率 {:.2%}。"
        ).format(
            total,
            before.get("active", 0),
            before.get("degraded", 0),
            before.get("offline", 0),
            after.get("active", 0),
            after.get("degraded", 0),
            after.get("offline", 0),
            float(summary.get("online_ratio_after") or 0.0),
        )
    ]
    restored = int(summary.get("restored_to_active") or 0)
    partial = int(summary.get("partially_recovered") or 0)
    if restored or partial:
        lines.append(f"恢复过程：{restored} 个原始基站恢复在线，{partial} 个离线基站恢复为降级可用。")
    for event in (summary.get("events") or [])[:6]:
        grid = event.get("grid") or {}
        step = event.get("recovery_step")
        step_text = f"step={step}" if step is not None else "step=--"
        lines.append(
            "恢复原始基站：{} grid=({}, {}) {} -> {} | {}。".format(
                event.get("label") or event.get("deployment_id") or "原始基站",
                grid.get("row", "--"),
                grid.get("col", "--"),
                event.get("from_status", "--"),
                event.get("to_status", "--"),
                step_text,
            )
        )
    remaining = max(0, len(summary.get("events") or []) - 6)
    if remaining:
        lines.append(f"还有 {remaining} 个原始基站恢复事件已写入部署后场景文件。")
    return lines


def _geo_bounds(region_grid) -> Optional[Dict[str, float]]:
    if not region_grid:
        return None
    return {
        "lat_min": float(region_grid.lat_min),
        "lat_max": float(region_grid.lat_max),
        "lon_min": float(region_grid.lon_min),
        "lon_max": float(region_grid.lon_max),
    }


def _node_geo_center(region_grid, row: int, col: int) -> Dict[str, float]:
    if not region_grid:
        return {}
    lat_min, lat_max, lon_min, lon_max = region_grid.cell_bounds(row, col)
    return {
        "lat": (float(lat_min) + float(lat_max)) / 2,
        "lon": (float(lon_min) + float(lon_max)) / 2,
    }


def _collect_deployed_station_nodes(report: Dict[str, Any], env) -> List[Dict[str, Any]]:
    stations: List[Dict[str, Any]] = []
    seen = set()

    for step in report.get("steps", []) or []:
        action_desc = step.get("action_desc") or {}
        location = action_desc.get("location")
        if not _valid_position(location):
            continue
        row, col = int(location[0]), int(location[1])
        comm_mode = action_desc.get("comm_mode")
        device = _resolve_deployment_device(env, comm_mode)
        base_key = action_desc.get("base_station") or device.get("base_station") or comm_mode
        base_label = action_desc.get("device_label") or device.get("device_label") or base_key
        coverage_radius = device.get("coverage_radius")
        coverage_radius_km = device.get("coverage_radius_km")

        dedupe_key = (row, col, str(base_key))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        stations.append(
            {
                "row": row,
                "col": col,
                "type": _base_station_node_type(base_key),
                "base_station": base_key,
                "device_type": action_desc.get("device_type") or device.get("device_type") or base_key,
                "device_label": base_label,
                "label": base_label,
                "mode": comm_mode,
                "coverage_radius": coverage_radius,
                "coverage_radius_km": coverage_radius_km,
            }
        )

    return stations


def _grid_shape(env) -> Tuple[int, int]:
    region_grid = getattr(env, "region_grid", None)
    if region_grid is not None:
        return int(region_grid.rows), int(region_grid.cols)
    if hasattr(env, "grid_rows") and hasattr(env, "grid_cols"):
        return int(getattr(env, "grid_rows")), int(getattr(env, "grid_cols"))
    size = int(getattr(env, "grid_size", 1))
    return size, size


def _grid_to_real_coords(row: int, col: int, rows: int, cols: int) -> Tuple[int, int]:
    x = int(round(((col + 0.5) / max(1, cols)) * REAL_MAP_WIDTH))
    y = int(round(((row + 0.5) / max(1, rows)) * REAL_MAP_HEIGHT))
    x = min(REAL_MAP_WIDTH, max(0, x))
    y = min(REAL_MAP_HEIGHT, max(0, y))
    return x, y


def _grid_to_scattered_real_coords(
    row: int,
    col: int,
    rows: int,
    cols: int,
    *,
    seed: Any,
    spread: float,
) -> Tuple[int, int]:
    center_x, center_y = _grid_to_real_coords(row, col, rows, cols)
    cell_width = REAL_MAP_WIDTH / max(1, cols)
    cell_height = REAL_MAP_HEIGHT / max(1, rows)
    effective_spread = max(0.0, min(1.72, float(spread)))
    angle = _stable_unit(f"{seed}:angle") * math.tau
    radius = (0.16 + math.sqrt(_stable_unit(f"{seed}:radius")) * 0.84) * effective_spread
    dx = math.cos(angle) * cell_width * 0.5 * radius
    dy = math.sin(angle) * cell_height * 0.5 * radius
    dx += (_stable_unit(f"{seed}:free-x") - 0.5) * cell_width * 0.22 * effective_spread
    dy += (_stable_unit(f"{seed}:free-y") - 0.5) * cell_height * 0.22 * effective_spread
    flow = (row * 1.371 + col * 0.917 + _stable_unit(f"{seed}:flow")) * math.pi
    dx += math.sin(flow) * cell_width * 0.08 * effective_spread
    dy += math.cos(flow * 0.83) * cell_height * 0.08 * effective_spread
    x = int(round(center_x + dx))
    y = int(round(center_y + dy))
    return min(REAL_MAP_WIDTH, max(0, x)), min(REAL_MAP_HEIGHT, max(0, y))


def _stable_unit(seed: Any) -> float:
    text = str(seed or "")
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    value = (value + ((value << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
    value ^= value >> 7
    value = (value + ((value << 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
    value ^= value >> 17
    value = (value + ((value << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (value & 0xFFFFFFFF) / 4294967296


def _base_station_node_type(base_key: Any) -> str:
    normalized = str(base_key or "").strip().lower()
    if "macro" in normalized or normalized.endswith("_enb"):
        return "MACRO_ENB"
    return "MANPACK_ENB"


def _valid_position(position: Any) -> bool:
    return isinstance(position, (list, tuple)) and len(position) >= 2


def _slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    return safe.strip("_") or "scene"


def _describe_scenario(env) -> Dict[str, Any]:
    scenario = getattr(env, "scenario", None)
    return {
        "name": getattr(scenario, "name", None) if scenario else None,
        "disaster_type": getattr(scenario, "disaster_type", None) if scenario else None,
        "num_users": getattr(env, "num_users", None),
        "max_steps": getattr(env, "max_steps", None),
        "reward_mode": getattr(env, "reward_mode", None),
        "reward_label": getattr(getattr(env, "reward_profile", None), "label", None)
        if getattr(env, "reward_profile", None)
        else None,
        "reward_description": getattr(getattr(env, "reward_profile", None), "description", None)
        if getattr(env, "reward_profile", None)
        else None,
        "evaluation_protocol": getattr(env, "evaluation_protocol", "standard"),
    }


def _capture_network_state(env, info: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = {
        "coverage_ratio": float(info.get("coverage_ratio", 0.0)),
        "broadcast_ratio": float(info.get("broadcast_ratio", 0.0)),
        "remaining_budget": float(info.get("remaining_budget", getattr(env, "remaining_budget", 0.0))),
        "total_users": getattr(env, "num_users", None),
        "avg_user_throughput": float(info.get("avg_user_throughput", 0.0)),
        "recent_throughput": float(info.get("recent_throughput", 0.0)),
        "device_cost": float(info.get("device_cost", 0.0)),
        "bandwidth_cost": float(info.get("bandwidth_cost", 0.0)),
        "reward_breakdown": info.get("reward_breakdown", {}),
        "evaluation_protocol": info.get("evaluation_protocol", getattr(env, "evaluation_protocol", "standard")),
        "residual_base_stations": info.get("residual_base_stations", []),
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
    region_grid = getattr(env, "region_grid", None)
    demands = getattr(env, "user_demands", None)
    connected = getattr(env, "user_connected", None)
    broadcast_served = getattr(env, "broadcast_served", None)
    custom_metadata = getattr(env, "custom_user_metadata", None)

    details: List[Dict[str, Any]] = []
    for idx in range(int(num_users)):
        entry: Dict[str, Any] = {"id": idx}
        if positions is not None and len(positions) > idx:
            coords = positions[idx]
            row = int(coords[0])
            col = int(coords[1])
            entry["position"] = (row, col)
            if region_grid:
                entry["region_id"] = region_grid.cell_index(row, col)
                entry["region_label"] = region_grid.cell_label(row, col)
                lat_min, lat_max, lon_min, lon_max = region_grid.cell_bounds(row, col)
                entry["lat_lon_bounds"] = {
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                }
        if demands is not None and len(demands) > idx:
            entry["demand"] = float(demands[idx])
        if connected is not None and len(connected) > idx:
            entry["connected"] = bool(connected[idx])
        if broadcast_served is not None and len(broadcast_served) > idx:
            entry["broadcast_served"] = bool(broadcast_served[idx])
        if custom_metadata is not None and len(custom_metadata) > idx and custom_metadata[idx]:
            entry.update(custom_metadata[idx])
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
    region_label = None
    if location and hasattr(env, "region_grid"):
        row, col = location
        region_label = env.region_grid.cell_label(row, col)

    comm_name = None
    if hasattr(env, "communication_modes") and len(env.communication_modes) > comm_idx:
        comm_name = env.communication_modes[comm_idx]

    broadcast_name = None
    if hasattr(env, "broadcast_modes") and len(env.broadcast_modes) > broadcast_idx:
        broadcast_name = env.broadcast_modes[broadcast_idx]

    device = _resolve_deployment_device(env, comm_name)
    return {
        "site_index": site_idx,
        "comm_index": comm_idx,
        "broadcast_index": broadcast_idx,
        "location": location,
        "region_label": region_label,
        "base_station": device.get("base_station"),
        "device_label": device.get("device_label"),
        "device_type": device.get("device_type"),
        "supported_modes": device.get("supported_modes", []),
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
    region_label = detail.get("region_label")

    if isinstance(idx, (int, np.integer)):
        id_text = f"{int(idx):02d}"
    else:
        id_text = str(idx) if idx is not None else "??"
    pos_text = f"({pos[0]}, {pos[1]})" if isinstance(pos, tuple) else str(pos) if pos is not None else "n/a"
    if region_label:
        pos_text = f"{pos_text} [{region_label}]"
    if isinstance(demand, (int, float, np.floating)):
        demand_text = f"{float(demand):.1f} Mbps"
    else:
        demand_text = "n/a"
    conn_text = "online" if connected else "offline"
    broadcast_text = "served" if broadcast_served else "unserved"
    return f"Device#{id_text} pos={pos_text} demand={demand_text} | connected={conn_text} | broadcast={broadcast_text}"
