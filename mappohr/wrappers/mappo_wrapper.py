import numpy as np
import gymnasium as gym
from luxai_s3.wrappers import LuxAIS3GymEnv
from common.environment import GameConstants, ActionType
from wrappers.utils.path_finding import DStarLitePlanner
from wrappers.static_planner_wrapper import SB3LuxEnvStaticPlanner
import os


class SB3LuxEnvMAPPO(SB3LuxEnvStaticPlanner):
    """
    A wrapper extending the static planner with MAPPO features including
    dynamic local planning with D* Lite and multi-agent coordination.

    This implementation incorporates the three components from the MAPPOH algorithm:
    1. Environment model
    2. Heuristic search planner (static global + dynamic local)
    3. Real-time planner based on MAPPO
    """

    def __init__(
        self,
        env=None,
        player_id="player_0",
        opponent_strategy="random",
        max_units=GameConstants.MAX_UNITS,
        replan_interval=None,  # How often to replan global paths
        look_ahead_steps=3,  # N steps to look ahead for collision detection
        use_heuristic_rules=True,
        model_dir="./training/ppo_lux_model_static_planner.zip",
        history_length=8,  # For recurrent policy
    ):
        super().__init__(env, player_id, opponent_strategy, max_units, replan_interval)

        self.max_units = max_units

        # Initialize the dynamic local planner using D* Lite
        self.dynamic_planner = DStarLitePlanner()

        # Additional MAPPO parameters
        self.look_ahead_steps = look_ahead_steps
        self.use_heuristic_rules = use_heuristic_rules

        # Track historical observations for RNN input
        self.history_length = history_length
        self.observation_history = []

        # For collision detection and coordination
        self.collision_risks = {}  # Track detected collision risks
        self.replan_decisions = {}  # Track when units decide to replan

        # Track previous actions for heuristic rules
        self.previous_actions = [-1] * max_units  # -1 means no previous action
        self.unit_states = {}  # Track states like "at_endpoint", "collided", etc.
        self.base_model_dir = model_dir

    def reset(self, **kwargs):
        """
        Reset the environment and initialize both static and dynamic planners.

        Args:
            **kwargs: Additional arguments for the base environment reset.

        Returns:
            obs: The initial observation after reset.
            info: Additional information about the environment state.
        """
        obs, info = super().reset(**kwargs)

        # Reset the dynamic planner
        self.dynamic_planner = DStarLitePlanner()

        # Reset history and states
        self.observation_history = [obs] * self.history_length
        self.previous_actions = [-1] * self.max_units
        self.unit_states = {}
        self.collision_risks = {}
        self.replan_decisions = {}

        # Immediately update the dynamic planner's cost map
        self.dynamic_planner.update_cost_map(obs)

        return obs, info

    def detect_collision_risk(self, obs, unit_id, action, look_ahead_steps):
        """
        Detect potential collisions within N steps ahead.

        Args:
            obs: The current observation from the environment.
            unit_id: The ID of the unit to check for collision risk.
            action: The action to simulate for the unit.
            look_ahead_steps: Number of steps to look ahead for collision detection.

        Returns:
            bool: True if a collision risk is detected, False otherwise.

        """
        player_idx = 0 if self.player_id == "player_0" else 1
        # Get the unit's current position
        unit_pos = tuple(obs["units_position"][player_idx][unit_id])

        # Get all other units' positions
        other_units = []
        for i in range(self.max_units):
            if i != unit_id and obs["units_mask"][player_idx][i] > 0:
                other_units.append((i, tuple(obs["units_position"][player_idx][i])))

        # Simulate movement for next N steps
        future_pos = self.simulate_future_position(unit_pos, action, look_ahead_steps)

        # Check for potential collisions
        for _, other_pos in other_units:
            if self.will_collide(future_pos, other_pos):
                return True

        return False

    def simulate_future_position(self, current_pos, action, steps):
        """
        Simulate future positions based on action and steps.

        Args:
            current_pos: The current position of the unit.
            action: The action to simulate.
            steps: Number of steps to simulate.

        Returns:
            tuple: The future position after simulating the action.
        """
        # Implementation to predict future positions
        future_pos = list(current_pos)
        for _ in range(steps):
            if action == ActionType.MOVE_UP.value:
                future_pos[1] -= 1
            elif action == ActionType.MOVE_DOWN.value:
                future_pos[1] += 1
            elif action == ActionType.MOVE_LEFT.value:
                future_pos[0] -= 1
            elif action == ActionType.MOVE_RIGHT.value:
                future_pos[0] += 1

            # Add boundary checks if necessary
            future_pos[0] = max(0, min(future_pos[0], GameConstants.MAP_WIDTH - 1))
            future_pos[1] = max(0, min(future_pos[1], GameConstants.MAP_HEIGHT - 1))

        return tuple(future_pos)

    def will_collide(self, pos1, pos2):
        """
        Check if two positions will collide.

        Args:
            pos1: The first position (x, y).
            pos2: The second position (x, y).

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        return pos1[0] == pos2[0] and pos1[1] == pos2[1]

    def has_reached_endpoint(self, unit_idx, obs):
        """
        Check if a unit has reached its target endpoint.

        Args:
            unit_idx: The index of the unit to check.
            obs: The current observation from the environment.

        Returns:
            bool: True if the unit has reached its endpoint, False otherwise.
        """
        player_id = 0 if self.player_id == "player_0" else 1
        if unit_idx not in self.path_planner.paths:
            return False

        if unit_idx not in self.path_planner.targets:
            return False

        unit_pos = tuple(obs["units_position"][player_id][unit_idx])
        target_pos = self.path_planner.targets[unit_idx]

        return unit_pos == target_pos

    def has_collided(self, unit_idx, obs):
        """
        Check if a unit has collided with an obstacle.

        Args:
            unit_idx: The index of the unit to check.
            obs: The current observation from the environment.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """

        player_id = 0 if self.player_id == "player_0" else 1

        if not obs["units_mask"][player_id][unit_idx]:
            return True  # Unit doesn't exist, might have been removed due to collision

        # Other collision conditions could be checked here
        return False

    def all_others_at_endpoint(self, unit_idx, obs):
        """
        Check if all other units have reached their endpoints.

        Args:
            unit_idx: The index of the unit to check.
            obs: The current observation from the environment.

        Returns:
            bool: True if all other units are at their endpoints, False otherwise.
        """
        player_id = 0 if self.player_id == "player_0" else 1

        for i in range(self.max_units):
            if i != unit_idx and obs["units_mask"][player_id][i] > 0:
                if not self.has_reached_endpoint(i, obs):
                    return False
        return True

    def is_current_path_longer(self, unit_idx):
        """
        Check if the current path is longer than the global guide path.

        Args:
            unit_idx: The index of the unit to check.

        Returns:
            bool: True if the current path is longer, False otherwise.
        """
        if (
            unit_idx not in self.path_planner.paths
            or unit_idx not in self.dynamic_planner.paths
        ):
            return False

        # Calculate lengths of both paths
        # This is an approximation since D* Lite doesn't store the full path
        static_goal = self.path_planner.paths[unit_idx]["goal"]
        dynamic_goal = self.dynamic_planner.paths[unit_idx]["goal"]

        if static_goal != dynamic_goal:
            return True

        # You might need a more sophisticated check here
        return False

    def all_others_waiting(self, obs, actions, unit_idx):
        """
        Check if all other units are waiting.

        Args:
            obs: The current observation from the environment.
            actions: The actions taken by all units.
            unit_idx: The index of the unit to check.
        Returns:
            bool: True if all other units are waiting, False otherwise.
        """
        player_id = 0 if self.player_id == "player_0" else 1

        for i in range(self.max_units):
            if i != unit_idx and obs["units_mask"][player_id][i] > 0:
                if i < len(actions) and actions[i] != 0:  # Not waiting
                    return False
        return True

    def apply_heuristic_rules(self, actions, obs):
        """
        Apply domain knowledge heuristic rules to actions.

        Args:
            actions: The actions taken by all units.
            obs: The current observation from the environment.

        Returns:
            modified_actions: The modified actions after applying heuristic rules.
        """
        modified_actions = actions.copy()
        processed_obs = self.base_wrapper.process_observation(
            obs, self.base_wrapper.last_info
        )
        player_idx = 0 if self.player_id == "player_0" else 1
        for unit_idx in range(min(len(actions), self.max_units)):
            # Skip if unit doesn't exist
            if (
                unit_idx >= len(processed_obs["units_mask"][player_idx])
                or not processed_obs["units_mask"][player_idx][unit_idx]
            ):
                continue

            # Rule 1: When agent reaches endpoint or collides, only "wait" is allowed
            if self.has_reached_endpoint(unit_idx, processed_obs) or self.has_collided(
                unit_idx, processed_obs
            ):
                modified_actions[unit_idx] = ActionType.STAY.value
                continue

            # Rule 2: All other agents at endpoint, replan if current path is longer than global
            if self.all_others_at_endpoint(unit_idx, processed_obs):
                if self.is_current_path_longer(unit_idx):
                    modified_actions[unit_idx] = ActionType.REPLAN.value
                    self.replan_decisions[unit_idx] = True
                else:
                    # Follow guide path
                    unit_pos = tuple(
                        processed_obs["units_position"][player_idx][unit_idx]
                    )
                    # Check if we have a static path for this unit
                    if unit_idx in self.path_planner.paths:
                        # Get the next action from the static planner's path
                        next_actions = self.path_planner.get_next_actions(
                            processed_obs, self.player_id
                        )

                        # Check if unit_idx is within the valid range of the actions list
                        if unit_idx < len(next_actions):
                            next_action = next_actions[unit_idx]
                        else:
                            next_action = ActionType.STAY.value

                        modified_actions[unit_idx] = next_action

                        # Clear any dynamic replanning for this unit
                        if unit_idx in self.dynamic_planner.paths:
                            # Remove the dynamic path to force using the static one
                            self.dynamic_planner.paths.pop(unit_idx, None)
                        continue

            unit_pos = tuple(processed_obs["units_position"][player_idx][unit_idx])

            # Rule 3: Check N steps forward, if no collision risk, follow guide path
            collision_risk = self.dynamic_planner.detect_collision_risk(
                processed_obs,
                unit_idx,
                self.simulate_future_position(unit_pos, actions[unit_idx], 1),
                self.look_ahead_steps,
            )

            if not collision_risk:
                # Follow guide path (no modification needed)
                self.collision_risks[unit_idx] = False
            else:
                self.collision_risks[unit_idx] = True

            # Rule 4: If collision risk in next step, forbid moving forward
            if self.collision_risks.get(unit_idx, False):
                next_pos = self.simulate_future_position(unit_pos, actions[unit_idx], 1)
                if next_pos != unit_pos:  # Only if it would actually move
                    modified_actions[unit_idx] = ActionType.STAY.value
                    continue

            # Rule 5: If previous action was "back" and still collision risk, move forward
            if ActionType.MOVE_UP.value <= self.previous_actions[
                unit_idx
            ] <= ActionType.MOVE_LEFT.value and self.collision_risks.get(
                unit_idx, False
            ):
                modified_actions[unit_idx] = actions[unit_idx]  # Allow current action
                continue

            # Rule 6: If all others wait, can move or replan
            if self.all_others_waiting(processed_obs, actions, unit_idx):
                # Allow current action (move or replan)
                if actions[unit_idx] != modified_actions[unit_idx]:
                    # Only restore movement and replan actions, not invalid ones
                    if (
                        ActionType.MOVE_UP.value
                        <= actions[unit_idx]
                        <= ActionType.MOVE_LEFT.value
                    ):
                        modified_actions[unit_idx] = actions[unit_idx]
                        continue

            # Rule 7: If action is SAP check if there are enemies near this relic node
            enemies_near_relic = False
            enemy_positions = []

            # First collect all enemy positions
            for enemy_idx in range(len(processed_obs["units_mask"][1])):
                if (
                    processed_obs["units_mask"][1][enemy_idx] > 0
                ):  # If enemy unit exists
                    enemy_pos = tuple(processed_obs["units_position"][1][enemy_idx])
                    enemy_positions.append(enemy_pos)

            # Then check if this unit is near a relic node
            near_relic = False
            relic_pos = None
            for j in range(len(processed_obs["relic_nodes_mask"])):
                if processed_obs["relic_nodes_mask"][j] == 1:
                    relic_pos = tuple(processed_obs["relic_nodes"][j])
                    relic_dist = abs(relic_pos[0] - unit_pos[0]) + abs(
                        relic_pos[1] - unit_pos[1]
                    )

                    if relic_dist <= 3:  # Unit is near relic
                        near_relic = True

                        # Check if any enemies are also near this relic
                        for enemy_pos in enemy_positions:
                            enemy_relic_dist = abs(relic_pos[0] - enemy_pos[0]) + abs(
                                relic_pos[1] - enemy_pos[1]
                            )
                            if enemy_relic_dist <= 4:  # Enemy is near the same relic
                                enemies_near_relic = True
                                break
                        break

            # Apply the appropriate strategy
            if near_relic:
                if actions[unit_idx] == ActionType.SAP.value:
                    if enemies_near_relic:
                        # Allow sapping when enemies are near relic nodes
                        pass
                    else:
                        if unit_idx in self.path_planner.paths:
                            # Get the next action from the static planner's path
                            next_action = self.path_planner.get_next_actions(
                                processed_obs, self.player_id
                            )

                            if unit_idx < len(next_action):
                                next_action = next_action[unit_idx]
                            else:
                                next_action = ActionType.STAY.value

                            # Get next movement from path planner to explore relic
                            modified_actions[unit_idx] = next_action
                        else:
                            # No path available, just wait
                            modified_actions[unit_idx] = ActionType.STAY.value

        return modified_actions

    def calculate_mappo_reward(self, actions, collision_detected):
        """
        Calculate the MAPPO reward based on actions and collision detection.

        Args:
            actions: The actions taken by all units.
            collision_detected: A dictionary indicating collision risks for each unit.

        Returns:
            mappo_reward: The calculated MAPPO reward.
        """

        # Apply MAPPO coordination penalties directly to the final reward
        mappo_reward = 0.0

        # Penalty 1: All robots execute "wait"
        if all(action == 0 for action in actions):
            mappo_reward -= 2.0

        # Penalty 2: All robots execute "replan"
        if all(action == 5 for action in actions):
            mappo_reward -= 2.0

        # Penalty 3: Collision risk with contradictory actions
        for unit_idx in range(min(len(actions), self.max_units)):
            for other_idx in range(min(len(actions), self.max_units)):
                if unit_idx != other_idx:
                    if (
                        unit_idx in collision_detected.keys()
                        and collision_detected[unit_idx]
                    ):  # Use the collision detection results from earlier
                        if (
                            actions[unit_idx]
                            in [
                                ActionType.MOVE_UP.value,
                                ActionType.MOVE_DOWN.value,
                                ActionType.MOVE_LEFT.value,
                                ActionType.MOVE_RIGHT.value,
                            ]
                            and actions[other_idx] == 5
                        ) or (
                            actions[other_idx]
                            in [
                                ActionType.MOVE_UP.value,
                                ActionType.MOVE_DOWN.value,
                                ActionType.MOVE_LEFT.value,
                                ActionType.MOVE_RIGHT.value,
                            ]
                            and actions[unit_idx] == 5
                        ):
                            mappo_penalty -= 1.5

        # Add coordination rewards (optional)
        mappo_reward = 0.0
        if hasattr(self, "previous_collision_detected") and any(
            self.previous_collision_detected
        ):
            if not any(
                collision_detected
            ):  # Successfully avoided previously detected collisions
                mappo_reward += 2.0

        return mappo_reward

    def step(self, actions):
        """
        Step the environment with enhanced MAPPO planning.

        Args:
            actions: Base actions from the RL agent

        Returns:
            observation, reward, terminated, truncated, info
        """

        models_dir = "../ppo_lux_model_mappohr/"
        model_files = (
            [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".zip")
            ]
            if os.path.exists(models_dir)
            else []
        )
        if len(model_files) > 0:
            self.base_model_dir = max(model_files, key=os.path.getmtime)

        # Get the current observation
        current_obs = self.base_wrapper.last_obs
        current_info = self.base_wrapper.last_info
        processed_current_obs = self.base_wrapper.process_observation(
            current_obs, current_info
        )
        player_idx = 0 if self.player_id == "player_0" else 1

        # Check for units that need to replan using D* Lite
        replan_units = []
        for unit_idx in range(min(len(actions), self.max_units)):
            # Check if unit exists and has a path
            if (
                processed_current_obs["units_mask"][player_idx][unit_idx] > 0
                and unit_idx in self.path_planner.paths
            ):
                # Detect collision risks
                unit_pos = tuple(
                    processed_current_obs["units_position"][player_idx][unit_idx]
                )
                next_pos = self.simulate_future_position(
                    unit_pos, unit_idx, actions[unit_idx]
                )

                if next_pos and self.dynamic_planner.detect_collision_risk(
                    processed_current_obs, unit_idx, next_pos, self.look_ahead_steps
                ):
                    replan_units.append(unit_idx)

        collision_detected = {}
        for unit_idx in range(min(len(actions), self.max_units)):
            if processed_current_obs["units_mask"][player_idx][unit_idx] > 0:
                unit_pos = tuple(
                    processed_current_obs["units_position"][player_idx][unit_idx]
                )
                next_pos = self.simulate_future_position(unit_pos, actions[unit_idx], 1)
                has_collision = self.dynamic_planner.detect_collision_risk(
                    processed_current_obs, unit_idx, next_pos, self.look_ahead_steps
                )
                collision_detected[unit_idx] = has_collision

        # Update the dynamic planner for units that need replanning
        for unit_idx in replan_units:
            # If action is replan or we detected a collision risk
            if actions[unit_idx] == ActionType.REPLAN or unit_idx in replan_units:
                unit_pos = tuple(
                    processed_current_obs["units_position"][player_idx][unit_idx]
                )
                target_pos = self.path_planner.targets.get(unit_idx, unit_pos)

                # Replan using D* Lite
                self.dynamic_planner.replan_path(
                    unit_idx,
                    processed_current_obs,
                    self.player_id,
                    unit_pos,
                    target_pos,
                )

                # Get the next action from the dynamic planner instead
                dynamic_action = self.dynamic_planner.get_next_action(unit_pos)
                if dynamic_action is not None:
                    actions[unit_idx] = dynamic_action
                else:
                    actions[unit_idx] = ActionType.STAY.value

        # Apply heuristic rules to modify actions
        if self.use_heuristic_rules:
            actions = self.apply_heuristic_rules(actions, current_obs)

        # Process and execute the modified actions using the parent's step method
        processed_obs, reward, terminated, truncated, info = super().step(
            actions, models_dir
        )

        # Update observation history for RNN
        self.observation_history.pop(0)
        self.observation_history.append(processed_obs)

        # Update previous actions
        self.previous_actions = list(actions)

        mappo_reward = self.calculate_mappo_reward(actions, collision_detected)
        final_reward = float(reward + mappo_reward)

        # Log to tensorboard
        lux_metrics_mappo = {}
        lux_metrics_mappo["mappo_reward"] = mappo_reward
        lux_metrics_mappo["final_mappo_reward"] = final_reward
        lux_metrics_mappo["collision_detected"] = sum(collision_detected.values())
        lux_metrics_mappo["replan_decisions"] = sum(self.replan_decisions.values())
        info["lux_metrics_mappo"] = lux_metrics_mappo

        return processed_obs, final_reward, terminated, truncated, info
