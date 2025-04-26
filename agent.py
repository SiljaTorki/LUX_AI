import os
import sys
import numpy as np
from stable_baselines3 import PPO
from common.environment import GameConstants, ActionType


def transform_obs(comp_obs, env_cfg=None, remainingOverageTime=60, team_id=0):
    """
    Transform observations to match the format expected by the PPO model.
    The model expects specific shapes for each observation field.
    """
    if "obs" in comp_obs:
        base_obs = comp_obs["obs"]
    else:
        base_obs = comp_obs

    flat_obs = {}

    # if "units" in base_obs:
    flat_obs["units_position"] = np.array(base_obs["units"]["position"], dtype=np.int32)
    flat_obs["units_energy"] = np.array(base_obs["units"]["energy"], dtype=np.int32)
    if flat_obs["units_energy"].ndim == 2:
        flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    flat_obs["units_mask"] = np.array(base_obs["units_mask"], dtype=np.int8)

    sensor_mask_arr = np.array(base_obs["sensor_mask"], dtype=np.int8)
    sensor_mask = sensor_mask_arr
    flat_obs["sensor_mask"] = sensor_mask

    # if "map_features" in base_obs:
    mf = base_obs["map_features"]
    flat_obs["map_features_tile_type"] = np.array(mf["tile_type"], dtype=np.int8)
    flat_obs["map_features_energy"] = np.array(mf["energy"], dtype=np.int8)
    flat_obs["relic_nodes_mask"] = np.array(base_obs["relic_nodes_mask"], dtype=np.int8)
    flat_obs["relic_nodes"] = np.array(base_obs["relic_nodes"], dtype=np.int32)
    flat_obs["team_points"] = np.array(base_obs["team_points"], dtype=np.int32)
    flat_obs["team_wins"] = np.array(base_obs["team_wins"], dtype=np.int32)
    flat_obs["steps"] = np.array([base_obs["steps"]], dtype=np.int32)
    flat_obs["match_steps"] = np.array([base_obs["match_steps"]], dtype=np.int32)

    flat_obs["env_cfg_map_width"] = np.array([env_cfg["map_width"]], dtype=np.int32)
    flat_obs["env_cfg_map_height"] = np.array([env_cfg["map_height"]], dtype=np.int32)
    flat_obs["env_cfg_max_steps_in_match"] = np.array(
        [env_cfg["max_steps_in_match"]], dtype=np.int32
    )
    flat_obs["env_cfg_unit_move_cost"] = np.array(
        [env_cfg["unit_move_cost"]], dtype=np.int32
    )
    flat_obs["env_cfg_unit_sap_cost"] = np.array(
        [env_cfg["unit_sap_cost"]], dtype=np.int32
    )
    flat_obs["env_cfg_unit_sap_range"] = np.array(
        [env_cfg["unit_sap_range"]], dtype=np.int32
    )
    flat_obs["remainingOverageTime"] = np.array([remainingOverageTime], dtype=np.int32)

    return flat_obs


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_last_directions = dict()
        self.unit_explore_locations = dict()
        model_path = os.path.join(os.path.dirname(__file__), "ppo_lux_model_base.zip")
        self.model = PPO.load(model_path)
        if "max_units" not in self.env_cfg:
            self.env_cfg["max_units"] = GameConstants.MAX_UNITS

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        # check if model is loaded
        if not hasattr(self, "model"):
            raise ValueError(
                "Model is not loaded. Please load the model before calling act."
            )

        flat_obs = transform_obs(obs, self.env_cfg, remainingOverageTime, self.team_id)
        if self.player == "player_1":
            flat_obs["units_position"] = np.array(
                [flat_obs["units_position"][1], flat_obs["units_position"][0]]
            )
            flat_obs["units_energy"] = np.array(
                [flat_obs["units_energy"][1], flat_obs["units_energy"][0]]
            )
            flat_obs["units_mask"] = np.array(
                [flat_obs["units_mask"][1], flat_obs["units_mask"][0]]
            )
        action, _ = self.model.predict(flat_obs, deterministic=False)
        action = np.array(action, dtype=np.int32)

        # Process actions for deployment
        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=np.int32)

        # Get sap range from config (default to 3 if not specified)
        sap_range = (
            self.env_cfg["unit_sap_range"] if "unit_sap_range" in self.env_cfg else 3
        )

        # Get our team's index (0 for player_0, 1 for player_1)
        team_idx = 0 if self.player == "player_0" else 1
        enemy_idx = 1 if team_idx == 0 else 0

        # Track our last movement direction for each unit to help with SAP targeting
        if (
            not hasattr(self, "unit_last_directions")
            or team_idx not in self.unit_last_directions
        ):
            self.unit_last_directions[team_idx] = [None] * max_units

        # Extract visible enemy positions
        visible_enemies = []
        for i in range(max_units):
            if flat_obs["units_mask"][enemy_idx][i]:
                enemy_pos = flat_obs["units_position"][enemy_idx][i]
                if -1 not in enemy_pos:  # Only include visible enemy units
                    visible_enemies.append((i, enemy_pos))

        # Extract visible relic nodes
        visible_relics = []
        for i in range(len(flat_obs["relic_nodes_mask"])):
            if flat_obs["relic_nodes_mask"][i]:
                relic_pos = flat_obs["relic_nodes"][i]
                if -1 not in relic_pos:  # Only include visible relics
                    visible_relics.append(relic_pos)
                    # Remember this relic for future use even when not visible
                    if not hasattr(self, "discovered_relics"):
                        self.discovered_relic_nodes_ids = []
                    if tuple(relic_pos) not in [
                        tuple(r) for r in self.discovered_relic_nodes_ids
                    ]:
                        self.discovered_relic_nodes_ids.append(relic_pos)

        # Process each unit's action
        for i, a in enumerate(action):
            # Get this unit's position
            unit_pos = flat_obs["units_position"][team_idx][i]

            if a <= 4:  # Movement or stay action
                actions[i, 0] = a
                actions[i, 1] = 0
                actions[i, 2] = 0

                # Remember movement direction for SAP targeting
                if 1 <= a <= 4:
                    self.unit_last_directions[team_idx][i] = a

            elif a == 5:  # SAP action
                actions[i, 0] = 5

                # Find best target for SAP
                best_target = None
                best_score = -1

                # Strategy 1: Target visible enemies
                for _, enemy_pos in visible_enemies:
                    dx = enemy_pos[0] - unit_pos[0]
                    dy = enemy_pos[1] - unit_pos[1]

                    # Check if in range
                    if abs(dx) <= sap_range and abs(dy) <= sap_range:
                        # Check for enemy clusters (higher value targets)
                        nearby_enemies = 0
                        for _, other_pos in visible_enemies:
                            if (
                                abs(other_pos[0] - enemy_pos[0]) <= 1
                                and abs(other_pos[1] - enemy_pos[1]) <= 1
                            ):
                                nearby_enemies += 1

                        # Score based on cluster size
                        score = nearby_enemies * 2
                        if score > best_score:
                            best_score = score
                            best_target = (dx, dy)

                # Strategy 2: Target relic nodes if no good enemy targets
                if best_target is None and visible_relics:
                    closest_dist = float("inf")
                    closest_relic = None

                    for relic_pos in visible_relics:
                        dist = abs(relic_pos[0] - unit_pos[0]) + abs(
                            relic_pos[1] - unit_pos[1]
                        )
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_relic = relic_pos

                    if closest_relic is not None:
                        dx = closest_relic[0] - unit_pos[0]
                        dy = closest_relic[1] - unit_pos[1]

                        # Clamp to sap range
                        dx = max(-sap_range, min(sap_range, dx))
                        dy = max(-sap_range, min(sap_range, dy))

                        best_target = (dx, dy)
                # Strategy 3: Target based on last movement direction
                if (
                    best_target is None
                    and self.unit_last_directions[team_idx][i] is not None
                ):
                    last_dir = self.unit_last_directions[team_idx][i]
                    dx, dy = 0, 0

                    if last_dir == 1:  # Up
                        dy = -sap_range
                    elif last_dir == 2:  # Right
                        dx = sap_range
                    elif last_dir == 3:  # Down
                        dy = sap_range
                    elif last_dir == 4:  # Left
                        dx = -sap_range

                    best_target = (dx, dy)

                # Strategy 4: Default targeting toward center
                if best_target is None:
                    map_width = flat_obs["env_cfg_map_width"]
                    map_height = flat_obs["env_cfg_map_height"]
                    center_x, center_y = map_width // 2, map_height // 2

                    dx = center_x - unit_pos[0]
                    dy = center_y - unit_pos[1]

                    # Clamp to sap range
                    dx = max(-sap_range, min(sap_range, dx))
                    dy = max(-sap_range, min(sap_range, dy))

                    best_target = (dx, dy)

                # Set the target coordinates
                actions[i, 1] = best_target[0]
                actions[i, 2] = best_target[1]

        return actions
