import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
import jax

from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from typing import Any
import dataclasses
import flax

from .params import EnvParams, env_params_ranges # Moved to project folder for ease of access.

class LuxEnvBase(gym.Env):
    """
    A wrapper for the Lux S3 env to make it compatible with SBX/Stable Baselines 3.
    This wrapper focuses on training a single player while allowing the opponent to be controlled
    by a different strategy.
    """

    def __init__(self, random_seed: int = 42):
        self.rng_key = jax.random.key(random_seed)          # Initializes Jax random number generator.
        self.jax_env = LuxAIS3Env(auto_reset=False)         # Sets the Jax Env to be the underlying Luxai env
        self.env_params: EnvParams = EnvParams()            # Default env parameters
        self.state = None                                   # Holds game state
        self.visited_mask = None                            # Track visited tiles for exploration bonus
        self.previous_obs = None

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()


    def _define_observation_space(self):
        """
        Observation Space Definition
        """

        # Define observation space components
        single_player_space = gym.spaces.Dict({
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

        obs_space = {
            'player_0': single_player_space,
            'player_1': single_player_space,
        }

        return gym.spaces.Dict(obs_space)

    def _define_action_space(self):
        """
        The agent's policy network should output actions conforming to this
        dictionary structure. The env's step function expects a dictionary
        containing actions for *both* players in this format and will internally
        combine these components into the (N, 3) format required by the
        underlying JAX env.

        Components:
        - 'action_type': MultiDiscrete([6] * max_units). The core action (0-5).
        - 'sap_target': Box(shape=(max_units, 2)). The relative dx, dy for sap.
        """


        single_player_action_space = Dict({
            "action_type": MultiDiscrete(
                nvec = [6] * self.env_params.max_units, # 6 possible action types (0-5) per unit
                dtype = np.int16
            ),
            "sap_target": Box(
                low = -max(env_params_ranges['unit_sap_range']),
                high = max(env_params_ranges['unit_sap_range']),
                shape = (self.env_params.max_units, 2),
                dtype = np.float32 # floats will be rounded and converted back to ints in the step function.
            )
        })

        action_space = {
            'player_0': single_player_action_space,
            'player_1': single_player_action_space,
        }

        return gym.spaces.Dict(action_space)



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

        self.previous_obs = None
        self.visited_mask = np.zeros(
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
            player_jax_action_internal = player_jax_action_internal.at[sap_mask, 1:].set(sap_targets_jax[sap_mask])

            action_jax[player] = player_jax_action_internal # Store the (N, 3) JAX array in the dict

        # Get RNG key for the step
        self.rng_key, step_key = jax.random.split(self.rng_key)

        previous_state_jax = self.state
        # Call the underlying JAX step function, and update the stored JAX state.
        obs_jax, state_jax, reward_jax, terminated_jax, truncated_jax, info_jax = self.jax_env.step(
            step_key, self.state, action_jax, self.env_params
        )
        self.state = state_jax

        # Convert JAX outputs to Numpy
        current_obs = to_numpy(flax.serialization.to_state_dict(obs_jax))
        reward = to_numpy(reward_jax) # Currently unused in reward shaping
        terminated = to_numpy(terminated_jax)
        truncated = to_numpy(truncated_jax)
        info = to_numpy(flax.serialization.to_state_dict(info_jax))

        points_delta_dict = {}
        for player_index, player in enumerate(['player_0', 'player_1']):
            previous_points = 0
            if self.previous_obs is not None:
                if player in self.previous_obs and 'team_points' in self.previous_obs[player]:
                    previous_points = self.previous_obs[player]['team_points'][player_index]
                    # try:
                    #     previous_points = self.previous_obs[player]['team_points'][player_index]
                    # except IndexError:
                    #     print(f"Warning: IndexError accessing previous points for {player}")
                    #     previous_points = 0 # Fallback
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

        previous_state_np = to_numpy(flax.serialization.to_state_dict(previous_state_jax))
        current_state_np = to_numpy(flax.serialization.to_state_dict(state_jax))

        shaped_rewards = self._calculate_reward(
            points_delta_dict=points_delta_dict,
            previous_state=previous_state_np,
            current_state=current_state_np,
        )

        # Ensure terminated and truncated are boolean dicts as expected by Gym.
        ## Not entirely sure if this is necessary.
        terminated_bool_dict = {k: bool(v) for k, v in terminated.items()}
        truncated_bool_dict = {k: bool(v) for k, v in truncated.items()}

        self.previous_obs = current_obs

        rewards = {
            "player_0": float(shaped_rewards.get('player_0', 0.0)),
            "player_1": float(shaped_rewards.get('player_1', 0.0))
        }
        terminateds = {
            "player_0": bool(terminated_bool_dict.get('player_0', False)),
            "player_1": bool(terminated_bool_dict.get('player_1', False)),
            "__all__": bool(terminated_bool_dict.get('__all__', False)) # Ensure __all__ is present if calculated earlier
        }
        truncateds = {
            "player_0": bool(truncated_bool_dict.get('player_0', False)),
            "player_1": bool(truncated_bool_dict.get('player_1', False)),
            "__all__": bool(truncated_bool_dict.get('__all__', False)) # Ensure __all__ is present if calculated earlier
        }
        # Ensure __all__ keys are correctly set based on individual agent statuses
        terminateds["__all__"] = terminateds["player_0"] or terminateds["player_1"]
        truncateds["__all__"] = truncateds["player_0"] or truncateds["player_1"]

        # Return as standard Gymnasium tuple
        return current_obs, rewards, terminateds, truncateds, info # Return dicts

    def _calculate_reward(self, points_delta_dict, previous_state, current_state) -> dict:
        point_delta_weight = 1.0
        win_bonus = 100.0

        final_rewards = {'player_0': 0.0, 'player_1': 0.0}

        # Check if a match ended THIS step by looking at the new state's match_steps
        # It resets to -1 internally, then increments to 0 for the next step start.
        match_just_ended = current_state.get('match_steps', -1) == 0

        p0_final_score = 0
        p1_final_score = 0
        if match_just_ended:
            try:
                # Use scores from the previous state (before match reset).
                p0_final_score = previous_state['team_points'][0]
                p1_final_score = previous_state['team_points'][0]
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Could not retrieve final scores from previous state for win bonus: {e}")
                match_just_ended = False # Cannot determine winner if scores unavailable


        # Calculate reward for each player
        for player_index, player in enumerate(['player_0', 'player_1']):
            # Reward for points gained this step
            points_reward = points_delta_dict[player] * point_delta_weight
            final_rewards[player] += points_reward

            # Win bonus if the match ended this step
            if match_just_ended:
                if player == 'player_0':
                    if p0_final_score > p1_final_score:
                        final_rewards[player] += win_bonus
                elif player == 'player_1':
                    if p1_final_score > p0_final_score:
                        final_rewards[player] += win_bonus
        return final_rewards

    def close(self):
        # Placeholder
        print(f"Environment closed.")
        pass

