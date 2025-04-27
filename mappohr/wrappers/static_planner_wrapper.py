import os
import gymnasium as gym
from luxai_s3.wrappers import LuxAIS3GymEnv
from common.environment import GameConstants
from wrappers.utils.path_finding import StaticPathPlanner
from wrappers.base_wrapper import SB3LuxEnvBase


class SB3LuxEnvStaticPlanner(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with SB3/Stable Baselines 3.
    This wrapper includes a static pathfinding algorithm for navigation.
    """

    def __init__(
        self,
        env=None,
        player_id="player_0",
        opponent_strategy="random",
        max_units=GameConstants.MAX_UNITS,
        replan_interval=None,  # How often to replan paths (in steps)
        model_dir="./training/ppo_lux_model_base.zip",
    ):
        if env is None:
            env = LuxAIS3GymEnv()
        super().__init__(env)

        # Path to model used for self-play
        self.base_model_dir = model_dir

        # Initialize the base wrapper
        self.base_wrapper = SB3LuxEnvBase(env, player_id, opponent_strategy, max_units)

        # Initialize the path planner
        self.path_planner = StaticPathPlanner()

        # Store parameters
        self.player_id = player_id
        self.replan_interval = replan_interval
        self.last_replan_step = -1
        self.current_step = 0

        # Define observation and action spaces from base wrapper
        self.observation_space = self.base_wrapper.observation_space
        self.action_space = self.base_wrapper.action_space

    def reset(self, **kwargs):
        """
        Reset the environment and initialize the path planner.

        Args:
            **kwargs: Additional arguments for the base environment reset.
        Returns:
            obs: The initial observation after reset.
            info: Additional information about the environment state.
        """

        obs, info = self.base_wrapper.reset(**kwargs)

        # Reset planning variables
        self.last_replan_step = 0
        self.current_step = 0

        # Reset path planner state
        self.path_planner.paths = {}  # Clear cached paths
        self.path_planner.targets = {}  # Clear targets

        # Initialize the cost map with the first observation
        self.path_planner.astar.update_cost_map(obs)

        # Immediately plan initial paths
        targets = self.path_planner.find_targets_for_units(obs, self.player_id)
        self.path_planner.compute_paths_for_all_units(obs, self.player_id, targets)

        return obs, info

    def is_near_relic_node(self, obs, unit_pos):
        """
        Check if a unit is near a relic node.

        Args:
            obs: The current observation from the environment.
            unit_pos: The position of the unit to check.

        Returns:
            bool: True if the unit is near a relic node, False otherwise.
        """

        # Check if the unit's position is within a certain distance from any relic node
        for relic_node in obs["relic_nodes"]:
            if (
                abs(unit_pos[0] - relic_node[0]) <= 1
                and abs(unit_pos[1] - relic_node[1]) <= 1
            ):
                return True
        return False

    def is_strategic_position(self, obs, unit_pos):
        """
        Check if a unit is in a strategic position.

        Args:
            obs: The current observation from the environment.
            unit_pos: The position of the unit to check.

        Returns:
            bool: True if the unit is in a strategic position, False otherwise.
        """
        enemy_start_pos = (
            (0, 0)
            if self.player_id == "player_1"
            else (GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1)
        )

        # Define strategic positions mid map, enemy base, relic nodes
        strategic_positions = [
            (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2),  # Mid map
            enemy_start_pos,  # Enemy base
        ]
        for relic_node in obs["relic_nodes"]:
            # Check if the relic node is within the map bounds
            if (
                relic_node[0] >= 0
                and relic_node[0] < GameConstants.MAP_WIDTH
                and relic_node[1] >= 0
                and relic_node[1] < GameConstants.MAP_HEIGHT
            ):
                # Add the relic node to strategic positions
                strategic_positions.append((relic_node[0], relic_node[1]))

        return unit_pos in strategic_positions

    def step(self, actions, models_dir=None):
        """
        Step the environment and use static path planning for unit movement.

        Args:
            actions: Actions from the agent, where action 5 represents a sap action

        Returns:
            observation, reward, terminated, truncated, info
        """

        # Get the current observation
        current_obs = self.base_wrapper.last_obs
        self.current_step += 1
        models_dir = (
            "../ppo_lux_model_static_planner/" if models_dir is None else models_dir
        )
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

        if current_obs is not None:
            processed_obs = self.base_wrapper.process_observation(
                current_obs, self.base_wrapper.last_info
            )
            self.current_step = processed_obs["steps"][0]

            # Only replan if frequency is set and enough steps have passed
            should_replan = (
                (
                    self.replan_interval is not None
                    and self.current_step - self.last_replan_step
                    >= self.replan_interval
                )
                or (self.replan_interval is None and self.current_step == 0)
                or self.path_planner.paths == {}
            )

            if should_replan:
                # Find targets for units
                targets = self.path_planner.find_targets_for_units(
                    processed_obs, self.player_id
                )

                # Compute paths for all units
                self.path_planner.compute_paths_for_all_units(
                    processed_obs, self.player_id, targets
                )

                # Update last replan step
                self.last_replan_step = self.current_step

            # Get the next actions for all units based on their paths
            path_actions = self.path_planner.get_next_actions(
                processed_obs, self.player_id
            )

            # Override with sap actions from the RL agent when appropriate
            unit_actions = []
            for unit_idx in range(min(len(path_actions), len(actions))):
                # Check if the RL agent wants to perform a sap action (action 5)
                if actions[unit_idx] == 5:
                    unit_actions.append(5)  # Use the sap action from the RL agent
                else:
                    unit_actions.append(path_actions[unit_idx])  # Use the path action

            # Use the actions from the RL agent if they are fewer than the number of units
            if len(unit_actions) < GameConstants.MAX_UNITS:
                unit_actions[len(unit_actions) :] = actions[len(unit_actions) :]

            # Step the environment with the computed actions
            obs, reward, terminated, truncated, info = self.base_wrapper.step(
                unit_actions, self.base_model_dir
            )

        else:
            # If no observation is available, just step the environment with the original actions
            obs, reward, terminated, truncated, info = self.base_wrapper.step(actions)

        # Add context-specific rewards (different from the base ones)
        contextual_sap_reward = 0.0
        player_idx = 0 if self.player_id == "player_0" else 1
        # Count sap actions that were taken in specific strategic contexts
        sap_count = 0
        for idx, action in enumerate(actions):
            if action == 5:  # Sap action
                sap_count += 1
                unit_pos = tuple(
                    (
                        obs["units_position"][player_idx][idx][0].item(),
                        obs["units_position"][player_idx][idx][1].item(),
                    )
                )
                # Add strategic context rewards
                if self.is_near_relic_node(obs, unit_pos):
                    # Extra reward for using sap near relic nodes (territory control)
                    contextual_sap_reward += 2.0

                if self.is_strategic_position(obs, unit_pos):
                    # Extra reward for using sap in positions of strategic importance
                    contextual_sap_reward += 2.0

        # Add these new contextual rewards
        enhanced_reward = reward + contextual_sap_reward

        # Log to tensorboard
        lux_metrics = {}
        lux_metrics["static_planner_bonus_sap_reward"] = contextual_sap_reward
        info["lux_metrics_static_planner"] = lux_metrics

        return obs, enhanced_reward, terminated, truncated, info
