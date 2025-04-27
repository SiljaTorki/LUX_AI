# Place this modified code in env/observation_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from collections import OrderedDict

class FlattenNormalizeObservation(gym.ObservationWrapper):
    """
    Wrapper to flatten the Lux AI S3 dictionary observation, normalize numerical values,
    and stack observations for both players into a single vector.

    The original observation space is a Dict {'player_0': p0_dict, 'player_1': p1_dict}.
    This wrapper converts it into a single flat Box space suitable for vectorized environments
    where one agent controls both players.

    Normalization is applied as before. The final output vector concatenates the
    processed observation for player_0 and player_1.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not hasattr(env, 'env_params'):
            raise ValueError("This wrapper requires the underlying env to have an 'env_params' attribute.")

        self.env_params = env.env_params
        self.original_space_dict_per_player = env.observation_space # Should be the Dict defined in LuxCustomGymEnv

        # Calculate the shape for a *single* player's flattened/normalized obs
        single_player_flat_size, self._single_player_flat_structure = self._calculate_single_player_flat_space()

        # The final observation space is the concatenation of both players' flattened obs
        total_flat_size = single_player_flat_size * 2 # Times two for player_0 and player_1
        # Define bounds (most are [-1, 1] or [0, 1] after normalization)
        # A simple Box from -1 to 1 covers most cases, can be refined if needed
        self.observation_space = Box(low=-1.0, high=1.0, shape=(total_flat_size,), dtype=np.float32)

    def _calculate_single_player_flat_space(self):
        """Calculates the size and structure of a single player's flattened obs."""
        flat_obs_dict = OrderedDict()
        T, N = self.original_space_dict_per_player['units']['position'].shape[:2]
        W = self.env_params.map_width
        H = self.env_params.map_height
        R = self.original_space_dict_per_player['relic_nodes_mask'].shape[0]

        # Sizes of each component after flattening
        flat_obs_dict['units_pos'] = T * N * 2
        flat_obs_dict['units_energy'] = T * N
        flat_obs_dict['units_mask'] = T * N
        flat_obs_dict['sensor_mask'] = W * H
        flat_obs_dict['map_energy'] = W * H
        flat_obs_dict['map_tile_type_onehot'] = W * H * 3
        flat_obs_dict['relic_nodes_mask'] = R
        flat_obs_dict['relic_nodes_pos'] = R * 2
        flat_obs_dict['team_points'] = T
        flat_obs_dict['team_wins'] = T
        flat_obs_dict['steps'] = 1
        flat_obs_dict['match_steps'] = 1

        total_size = sum(flat_obs_dict.values())
        return total_size, flat_obs_dict # Return size and structure


    def _process_single_player_obs(self, player_obs):
        """Helper function to flatten and normalize a single player's observation dict."""
        flat_parts = []
        W = self.env_params.map_width
        H = self.env_params.map_height
        width_divisor = max(1, W - 1)
        height_divisor = max(1, H - 1)

        # Units - Position
        units_pos = player_obs['units']['position'].astype(np.float32)
        mask_pos = (units_pos == -1)
        units_pos[:, :, 0] = units_pos[:, :, 0] / width_divisor
        units_pos[:, :, 1] = units_pos[:, :, 1] / height_divisor
        units_pos[mask_pos] = -1.0
        flat_parts.append(units_pos.flatten())

        # Units - Energy
        units_energy = player_obs['units']['energy'].astype(np.float32)
        mask_energy = (units_energy == -1)
        units_energy = np.clip(units_energy, 0, self.env_params.max_unit_energy) / max(1, self.env_params.max_unit_energy)
        units_energy[mask_energy] = -1.0
        flat_parts.append(units_energy.flatten())

        # Units Mask
        flat_parts.append(player_obs['units_mask'].astype(np.float32).flatten())

        # Sensor Mask
        flat_parts.append(player_obs['sensor_mask'].astype(np.float32).flatten())

        # Map Features - Energy
        map_energy = player_obs['map_features']['energy'].astype(np.float32)
        mask_map_energy = (map_energy == -1)
        # Clip to 0 before normalizing energy per tile
        map_energy = np.clip(map_energy, 0, self.env_params.max_energy_per_tile) / max(1, self.env_params.max_energy_per_tile)
        map_energy[mask_map_energy] = -1.0
        flat_parts.append(map_energy.flatten())

        # Map Features - Tile Type -> One-Hot
        tile_type = player_obs['map_features']['tile_type']
        one_hot = np.zeros((W, H, 3), dtype=np.float32)
        one_hot[..., 0] = (tile_type == 0) # Empty
        one_hot[..., 1] = (tile_type == 1) # Nebula
        one_hot[..., 2] = (tile_type == 2) # Asteroid
        flat_parts.append(one_hot.reshape(-1))

        # Relic Nodes Mask
        flat_parts.append(player_obs['relic_nodes_mask'].astype(np.float32).flatten())

        # Relic Nodes Position
        relic_pos = player_obs['relic_nodes'].astype(np.float32)
        mask_relic_pos = (relic_pos == -1)
        relic_pos[:, 0] = relic_pos[:, 0] / width_divisor
        relic_pos[:, 1] = relic_pos[:, 1] / height_divisor
        relic_pos[mask_relic_pos] = -1.0
        flat_parts.append(relic_pos.flatten())

        # Team Points (Heuristic Max)
        # Max points aren't strictly bounded. Using a heuristic might be okay.
        max_possible_points = (self.env_params.max_steps_in_match + 1) * self.env_params.max_units
        team_points = player_obs['team_points'].astype(np.float32)
        team_points = np.clip(team_points, 0, max_possible_points) / max(1, max_possible_points)
        flat_parts.append(team_points.flatten())

        # Team Wins
        team_wins = player_obs['team_wins'].astype(np.float32) / max(1, self.env_params.match_count_per_episode)
        flat_parts.append(team_wins.flatten())

        # Steps
        total_max_steps = (self.env_params.max_steps_in_match + 1) * self.env_params.match_count_per_episode
        steps = np.array([player_obs['steps']], dtype=np.float32) / max(1, total_max_steps)
        flat_parts.append(steps.flatten())

        # Match Steps
        match_steps = np.array([player_obs['match_steps']], dtype=np.float32) / max(1, self.env_params.max_steps_in_match)
        flat_parts.append(match_steps.flatten())

        # Concatenate
        flat_obs = np.concatenate(flat_parts)
        return flat_obs.astype(np.float32)


    def observation(self, obs):
        """
        Processes the raw observation dictionary {'player_0': p0_obs, 'player_1': p1_obs}
        and returns a single stacked numpy array concatenating the processed observations
        for both players.
        """
        assert isinstance(obs, dict), "Input observation must be a dictionary."
        assert "player_0" in obs and "player_1" in obs, "Observation dictionary missing player keys."

        # Process each player's observation dictionary
        flat_obs_p0 = self._process_single_player_obs(obs["player_0"])
        flat_obs_p1 = self._process_single_player_obs(obs["player_1"])

        # Concatenate the flattened observations
        stacked_obs = np.concatenate([flat_obs_p0, flat_obs_p1])

        # Sanity check the final shape
        assert stacked_obs.shape == self.observation_space.shape, \
            f"Shape mismatch for stacked obs: expected {self.observation_space.shape}, got {stacked_obs.shape}"

        return stacked_obs