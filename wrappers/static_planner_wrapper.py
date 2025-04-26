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
        model_dir="ppo_lux_model_base.zip",
    ):
        if env is None:
            env = LuxAIS3GymEnv()
        super().__init__(env)

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
        """Reset the environment and replan paths."""
        obs, info = self.base_wrapper.reset(**kwargs)

        # Reset planning variables
        self.last_replan_step = 0
        self.current_step = 0

        # Reset path planner state
        self.path_planner.paths = {}  # Clear cached paths
        self.path_planner.targets = {}  # Clear targets

        # You might also want to initialize the cost map with the first observation
        self.path_planner.astar.update_cost_map(obs)

        # Immediately plan initial paths
        targets = self.path_planner.find_targets_for_units(obs, self.player_id)
        self.path_planner.compute_paths_for_all_units(obs, self.player_id, targets)

        return obs, info

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
        models_dir = "./ppo_lux_model_static_planner/" if models_dir is None else models_dir
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

            # Check if we need to replan (first step or replan interval)
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
            # Here we assume actions is a MultiDiscrete space with 6 possible actions per unit
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

        return obs, reward, terminated, truncated, info