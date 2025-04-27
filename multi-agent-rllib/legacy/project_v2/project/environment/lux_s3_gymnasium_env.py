import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
import jax
from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from typing import Any, SupportsFloat
import dataclasses
import flax

from .params import EnvParams, env_params_ranges # Moved to project folder for ease of access.


class LuxCustomGymEnv(gym.Env):
    def __init__(self, random_seed: int = 42):
        self.rng_key = jax.random.key(random_seed)          # Initializes Jax random number generator.
        self.jax_env = LuxAIS3Env(auto_reset=False)         # Sets the Jax Env to be the underlying Luxai env
        self.env_params: EnvParams = EnvParams()            # Default env parameters
        self.state = None                                   # Holds game state
        self.visited_mask = None                            # Track visited tiles for exploration bonus
        self.previous_obs = None

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()


    def _define_action_space(self):
        """
        Defines the action space for a SINGLE PLAYER as a Dict space.

        The agent's policy network should output actions conforming to this
        dictionary structure. The env's step function expects a dictionary
        containing actions for *both* players in this format and will internally
        combine these components into the (N, 3) format required by the
        underlying JAX env.

        Components:
        - 'action_type': MultiDiscrete([6] * max_units). The core action (0-5).
        - 'sap_target': Box(shape=(max_units, 2)). The relative dx, dy for sap.
        """


        player_action_space = Dict({
            "action_type": MultiDiscrete(
                nvec = [6] * self.env_params.max_units, # 6 possible action types (0-5) per unit
                dtype = np.int16
            ),
            "sap_target": Box(
                low = -max(env_params_ranges['unit_sap_range']),
                high = max(env_params_ranges['unit_sap_range']),
                shape = (self.env_params.max_units, 2),
                dtype = np.int16 # JAX env expects int16 usually
            )
        })
        return player_action_space


    def _define_observation_space(self):
        """
        Observation Space Definition
        """

        # Define observation space components
        obs_space = gym.spaces.Dict({
            "units": gym.spaces.Dict({
                # Position: (x, y). Masked with -1. Coords from 0 to W-1 or H-1.
                "position": gym.spaces.Box(
                    low=-1, high=max(self.env_params.map_width, self.env_params.map_height) - 1,
                    shape=(self.env_params.num_teams, self.env_params.max_units, 2), dtype=np.int16),

                # Energy: Masked with -1. Bounds from fixed EnvParams.
                "energy": gym.spaces.Box(
                    low=-1, high=self.env_params.max_unit_energy,
                    shape=(self.env_params.num_teams, self.env_params.max_units, 1), dtype=np.int16)
            }),

            "units_mask": gym.spaces.MultiBinary((self.env_params.num_teams, self.env_params.max_units)),   # Boolean indicating visibility/existence.
            "sensor_mask": gym.spaces.MultiBinary((self.env_params.map_width, self.env_params.map_height)), # Boolean indicating tile visibility.
            "map_features": gym.spaces.Dict({
                "energy": gym.spaces.Box(
                    low=float(self.env_params.min_energy_per_tile), high=float(self.env_params.max_energy_per_tile),
                    shape=(self.env_params.map_width, self.env_params.map_height), dtype=np.float32),

                # Tile Type: Masked with -1. 0=empty, 1=nebula, 2=asteroid.
                "tile_type": gym.spaces.Box(
                    low=-1, high=2, shape=(self.env_params.map_width, self.env_params.map_height), dtype=np.int8)
            }),

            # Relic Nodes Mask: Boolean indicating visibility/existence.
            "relic_nodes_mask": gym.spaces.MultiBinary(self.env_params.max_relic_nodes),

            # Relic Nodes Position: (x, y). Masked with -1. Coords from 0 to W-1 or H-1.
            "relic_nodes": gym.spaces.Box(
                low=-1, high=max(self.env_params.map_width, self.env_params.map_height) - 1,
                shape=(self.env_params.max_relic_nodes, 2), dtype=np.int16),

            # Team Points: Min 0. Max points unknown, using inf.
            "team_points": gym.spaces.Box(low=0, high=np.inf, shape=(self.env_params.num_teams,), dtype=np.int32),

            # Team Wins: Min 0, Max defined by fixed EnvParams.
            "team_wins": gym.spaces.Box(
                low=0, high=self.env_params.match_count_per_episode,
                shape=(self.env_params.num_teams,), dtype=np.int8),

            # Global Step Count: Min 0. Max could be match_count * max_steps? Using inf for safety.
            "steps": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),

            # Match Step Count: Min 0, Max defined by fixed EnvParams.
            "match_steps": gym.spaces.Box(low=0, high=self.env_params.max_steps_in_match, shape=(), dtype=np.int16)
        })

        return obs_space

    def render(self):
        self.jax_env.render(self.state, self.env_params)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None ) -> tuple[Any, dict[str, Any]]:
        """
        Resets the env for a new episode.
        Args:
            seed (int | None): Seed for the random number generator. If None, uses the
                               existing sequence.
            options (dict[str, Any] | None): Dictionary of options. Can include
                                             `{"params": EnvParams(...) }` to override
                                             default or randomized parameters.

        Returns:
            tuple[Any, dict[str, Any]]: A tuple containing the initial observation
                                        and the info dictionary.
        """
        # Reseed if provided, and split off new key for future use.
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)

        # Randomize game parameters based on env_params_ranges by sampling from the param ranges.
        randomized_game_params = dict()
        for key, value in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[key] = jax.random.choice(subkey, jax.numpy.array(value)).item()

        # Create EnvParams instance with defaults overridden by randomized values
        params = EnvParams(**randomized_game_params)

        # Override parameters from options if present.
        if options is not None and "params" in options:
            params = options["params"]

        # Store the final EnvParams for this episode, define action and observation spaces based on final params
        self.env_params = params
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        # Reset the underlying JAX env, get initial observations(converted to numpy from jax) and game state
        obs, self.state = self.jax_env.reset(reset_key, params=self.env_params)
        obs = to_numpy(flax.serialization.to_state_dict(obs))

        self.previous_obs = None # Reset previous observation tracker
        self.visited_mask = np.zeros( # Also reset visited mask
            (self.env_params.map_width, self.env_params.map_height), dtype=bool
        )

        # Define the info dictionary and create parameter subset of only the agent-visible parameters
        params_dict = dataclasses.asdict(self.env_params)
        params_dict_kept = dict()
        params_to_keep = [
            "max_units", "match_count_per_episode", "max_steps_in_match",
            "map_height", "map_width", "num_teams", "unit_move_cost",
            "unit_sap_cost", "unit_sap_range", "unit_sensor_range",
        ]
        for key in params_to_keep:
            params_dict_kept[key] = params_dict[key]

        self.previous_obs = obs

        return obs, dict(
            params=params_dict_kept,    # Agent-visible params
            full_params=params_dict,    # All params for debugging/analysis
            state=self.state            # Initial JAX state
        )


    def step(self, action: Any) -> tuple[Any, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[str, Any]]:
        """
        Advances the env by one time step using the provided actions.

        Args:
            action (Any): A dictionary containing actions for both players,
                          structured according to the Dict action space defined in
                          _define_action_space(). Expected format:
                          {'player_0': {'action_type': ndarray(N,), 'sap_target': ndarray(N,2)},
                           'player_1': {'action_type': ndarray(N,), 'sap_target': ndarray(N,2)}}

        Returns:
            tuple[Any, dict[Any, float], bool, bool, dict[str, Any]]:
                A tuple containing (observation, reward, terminated, truncated, info).
                - observation: NumPy array or dict of arrays.
                - reward: dict {'player_0': r0, 'player_1': r1}.
                - terminated: dict {'player_0': bool, 'player_1': bool}.
                - truncated: dict {'player_0': bool, 'player_1': bool}.
                - info: Dictionary with auxiliary information (converted to NumPy).
        """
        action_jax = {}
        for player in ["player_0", "player_1"]:
            player_action_dict = action[player]
            action_types = player_action_dict["action_type"]
            sap_targets = player_action_dict["sap_target"]

            # Convert individual components to JAX arrays
            action_types_jax = jax.numpy.asarray(action_types, dtype=jax.numpy.int16)
            sap_targets_jax = jax.numpy.asarray(sap_targets, dtype=jax.numpy.int16)

            # Reconstruct the (N, 3) structure expected *inside* the dict value by LuxAIS3Env
            player_jax_action_internal = jax.numpy.zeros((self.env_params.max_units, 3), dtype=jax.numpy.int16)
            player_jax_action_internal = player_jax_action_internal.at[:, 0].set(action_types_jax)

            # Only fill sap targets if action type is 5
            sap_mask = (action_types_jax == 5)
            # Use .at[...].set(...) for JAX compatibility if needed, though direct slicing might work if sap_targets_jax has the right shape
            # Ensure shapes align: sap_targets_jax[sap_mask] should be (num_saps, 2)
            player_jax_action_internal = player_jax_action_internal.at[sap_mask, 1:].set(sap_targets_jax[sap_mask])

            action_jax[player] = player_jax_action_internal # Store the (N, 3) JAX array in the dict

        # Get RNG key for the step
        self.rng_key, step_key = jax.random.split(self.rng_key)

        # Call the underlying JAX step function, and update the stored JAX state.
        obs_jax, state_jax, reward_jax, terminated_jax, truncated_jax, info_jax = self.jax_env.step(
            step_key, self.state, action_jax, self.env_params
        )
        self.state = state_jax

        # Convert JAX outputs to Numpy
        current_obs = to_numpy(flax.serialization.to_state_dict(obs_jax))
        reward = to_numpy(reward_jax)
        terminated = to_numpy(terminated_jax)
        truncated = to_numpy(truncated_jax)
        info = to_numpy(flax.serialization.to_state_dict(info_jax))


        points_delta_dict = {}
        for player_index, player in enumerate(['player_0', 'player_1']):
            previous_points = 0
            if self.previous_obs is not None:
                # Ensure indexing matches the shape of team_points (T,)
                if player in self.previous_obs and 'team_points' in self.previous_obs[player]:
                    try:
                        previous_points = self.previous_obs[player]['team_points'][player_index]
                    except IndexError:
                        print(f"Warning: IndexError accessing previous points for {player}")
                        previous_points = 0 # Fallback
                else:
                    # Handle case where previous_obs might be missing structure
                    # This can happen on the very first step after reset if not handled carefully
                    previous_points = 0

            current_points = 0
            if player in current_obs and 'team_points' in current_obs[player]:
                try:
                    current_points = current_obs[player]['team_points'][player_index]
                except IndexError:
                    print(f"Warning: IndexError accessing current points for {player}")
                    current_points = 0 # Fallback
            else:
                print(f"Warning: Could not find points for {player} in current_obs_np")


            points_delta_dict[player] = max(0, current_points - previous_points) # Reward only positive changes

        shaped_rewards = self._calculate_reward(current_obs, points_delta_dict, terminated, truncated, info)


        # Ensure terminated and truncated are boolean dicts as expected by Gym.
        # Not entirely sure if this is necessary.
        terminated_bool_dict = {k: bool(v) for k, v in terminated.items()}
        truncated_bool_dict = {k: bool(v) for k, v in truncated.items()}

        self.previous_obs = current_obs

        # Return as standard Gymnasium tuple
        return current_obs, shaped_rewards, terminated_bool_dict, truncated_bool_dict, info

    def _calculate_reward(self, current_obs, points_delta_dict, terminated, truncated, info) -> dict:
        """
        Calculates rewards based on point changes and shaping terms.

        Args:
            current_obs (dict): The observation dictionary for the current step (numpy).
            points_delta_dict (dict): Dictionary mapping player_id to points gained this step.
            terminated (dict): Termination flags.
            truncated (dict): Truncation flags.
            info (dict): Info dictionary (may contain raw JAX state under 'state' key).

        Returns:
            dict: Dictionary mapping player_id to calculated reward.
        """
        final_rewards = {}

        # --- Reward Weights ---
        point_delta_weight = 5.0    # Reward for scoring points
        exploration_weight = 0.01   # Reward for visiting new tiles
        relic_prox_weight = 0.02    # Reward for being near visible relics
        cluster_penalty_weight = -0.005 # Negative weight for penalty

        # Update visited mask globally first (using player 0's sensor mask for simplicity,
        # assumes symmetry or doesn't matter which player reveals)
        if 'player_0' in current_obs:
            p0_sensor_mask = current_obs['player_0'].get('sensor_mask', None)
            if p0_sensor_mask is not None and self.visited_mask is not None:
                # Ensure masks are compatible shapes before OR operation
                if p0_sensor_mask.shape == self.visited_mask.shape:
                    newly_visited_global = p0_sensor_mask & (~self.visited_mask)
                    self.visited_mask |= p0_sensor_mask # Update global visited mask
                else:
                    # Handle shape mismatch
                    newly_visited_global = np.zeros_like(p0_sensor_mask, dtype=bool)


            else:
                newly_visited_global = np.zeros((self.env_params.map_width, self.env_params.map_height), dtype=bool) # Default if no mask
        else:
            newly_visited_global = np.zeros((self.env_params.map_width, self.env_params.map_height), dtype=bool)


        for player_index, player in enumerate(['player_0', 'player_1']):
            if player not in current_obs:
                final_rewards[player] = 0.0
                continue # Skip if player data isn't in observation

            player_obs = current_obs[player]
            points_delta = points_delta_dict.get(player, 0) # Points gained this step

            shaped_reward = 0.0

            # --- Exploration Bonus ---
            # Use the globally calculated newly visited tiles count
            exploration_bonus = np.sum(newly_visited_global) * exploration_weight
            shaped_reward += exploration_bonus

            # --- Player Specific Calculations ---
            try:
                # Safely access observation components
                player_units_mask = player_obs.get('units_mask', np.zeros((self.env_params.num_teams, self.env_params.max_units), dtype=bool))
                player_units_data = player_obs.get('units', {})
                player_unit_positions_all = player_units_data.get('position', np.full((self.env_params.num_teams, self.env_params.max_units, 2), -1, dtype=np.int16))

                # Ensure we are accessing the correct player's data
                if player_index < player_units_mask.shape[0]:
                    active_unit_indices = np.where(player_units_mask[player_index])[0]
                    num_active_units = len(active_unit_indices)
                else:
                    active_unit_indices = []
                    num_active_units = 0

                if num_active_units > 0 and player_index < player_unit_positions_all.shape[0]:
                    active_unit_positions = player_unit_positions_all[player_index][active_unit_indices]

                    # Filter out invalid positions (e.g., -1)
                    valid_pos_mask = np.all(active_unit_positions != -1, axis=1)
                    active_unit_positions = active_unit_positions[valid_pos_mask]
                    num_active_units = len(active_unit_positions) # Update count

                    if num_active_units > 0:
                        # --- Relic Proximity Bonus ---
                        visible_relics_mask = player_obs.get('relic_nodes_mask', np.zeros(self.env_params.max_relic_nodes, dtype=bool))
                        relic_nodes_pos = player_obs.get('relic_nodes', np.full((self.env_params.max_relic_nodes, 2), -1, dtype=np.int16))

                        visible_relic_indices = np.where(visible_relics_mask)[0]
                        relic_proximity_bonus = 0.0
                        if len(visible_relic_indices) > 0:
                            visible_relic_positions = relic_nodes_pos[visible_relic_indices]
                            # Filter out invalid relic positions
                            valid_relic_mask = np.all(visible_relic_positions != -1, axis=1)
                            visible_relic_positions = visible_relic_positions[valid_relic_mask]

                            if len(visible_relic_positions) > 0:
                                # Calculate min distance from each unit to *any* visible relic
                                all_distances = np.abs(active_unit_positions[:, np.newaxis, :] - visible_relic_positions[np.newaxis, :, :]).sum(axis=2)
                                min_dist_per_unit = np.min(all_distances, axis=1)
                                proximity_reward = 1.0 / np.maximum(1, min_dist_per_unit) # Avoid division by zero
                                relic_proximity_bonus = np.sum(proximity_reward) * relic_prox_weight

                        shaped_reward += relic_proximity_bonus

                        # --- Clustering Penalty ---
                        clustering_penalty = 0.0
                        if num_active_units > 1:
                            unit_pos_diff = np.abs(active_unit_positions[:, np.newaxis, :] - active_unit_positions[np.newaxis, :, :])
                            pairwise_distances = unit_pos_diff.sum(axis=2)
                            too_close_threshold = 1
                            close_pairs = pairwise_distances <= too_close_threshold
                            np.fill_diagonal(close_pairs, False)
                            num_close_pairs = np.sum(close_pairs) / 2
                            clustering_penalty = num_close_pairs * cluster_penalty_weight # weight is negative
                        shaped_reward += clustering_penalty

            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Error processing shaping components for {player}: {e}")
                pass # Continue without these shaping terms if data is missing


            # --- Combine Rewards ---
            # Primary reward is the points delta, modified by shaping terms
            final_rewards[player] = (points_delta * point_delta_weight) + shaped_reward

        return final_rewards

    def close(self):
        # Placeholder
        print(f"Environment closed.")
        pass
