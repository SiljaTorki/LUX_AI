import os
import sys
import torch
import numpy as np

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns

from params import EnvParams

LSTM_CELL_SIZE = 256

class Agent():
    # __init__ method remains the same as the previous version
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.policy_id = "p0" if self.player == "player_0" else "p1"
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.env_params = EnvParams

        print(f"Player {self.player} (Team ID: {self.team_id}, Policy: {self.policy_id}): Attempting to load RLlib RLModule...", file=sys.stderr)
        base_algo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "saved_algos_rllib", "base_algo"))
        rl_module_checkpoint_path = os.path.join(
            base_algo_dir, "learner_group", "learner", "rl_module", self.policy_id,
        )

        self.rl_module = RLModule.from_checkpoint(rl_module_checkpoint_path)
        self.lstm_cell_size = LSTM_CELL_SIZE
        self.init_state = self.state = self.rl_module.get_initial_state()


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        flat_obs = self._process_single_player_obs(obs)
        input_dict = {Columns.OBS: torch.from_numpy(flat_obs).unsqueeze(0), Columns.STATE_IN: self.state}
        rl_module_out = self.rl_module.forward_inference(input_dict)

        print(rl_module_out)

        ## NEED TO PROCESS THE RL MODULE OUT STUFF SO IT ABIDES WITH THE BELOW ASSERT
        actions = np.zeros((16, 3), dtype=int)
        assert isinstance(actions, np.ndarray) and actions.shape == (16, 3), f"Structure mismatch: 'actions' should be a NumPy array with shape (16, 3), but got type {type(actions)} and shape {actions.shape}"

        return actions


    def _process_single_player_obs(self, player_obs):
        """Helper function to flatten and normalize a single player's observation dict."""
        flat_parts = []
        W = self.env_params.map_width
        H = self.env_params.map_height
        width_divisor = max(1, W - 1)
        height_divisor = max(1, H - 1)

        # --- Normalization ---
        units_pos = player_obs['units']['position'].astype(np.float32)
        mask_pos = (units_pos == -1)
        units_pos[:, :, 0] = units_pos[:, :, 0] / width_divisor
        units_pos[:, :, 1] = units_pos[:, :, 1] / height_divisor
        units_pos[mask_pos] = -1.0
        flat_parts.append(units_pos.flatten())

        # Units - Energy
        units_energy = player_obs['units']['energy'].astype(np.float32)
        mask_energy = (units_energy == -1)
        # Ensure energy value does not exceed max before division
        units_energy = np.clip(units_energy, 0, self.env_params.max_unit_energy) / max(1, self.env_params.max_unit_energy)
        units_energy[mask_energy] = -1.0
        flat_parts.append(units_energy.flatten()) # Flatten the (T, N, 1) array

        # Units Mask
        flat_parts.append(player_obs['units_mask'].astype(np.float32).flatten())

        # Sensor Mask
        flat_parts.append(player_obs['sensor_mask'].astype(np.float32).flatten())

        # Map Features - Energy
        map_energy = player_obs['map_features']['energy'].astype(np.float32)
        mask_map_energy = (map_energy == -1) # Assuming -1 is mask for non-visible map energy

        # Normalize between min and max energy per tile, scaling to [-1, 1] or [0, 1]
        energy_range = max(1, self.env_params.max_energy_per_tile - self.env_params.min_energy_per_tile)
        map_energy = np.clip(map_energy, self.env_params.min_energy_per_tile, self.env_params.max_energy_per_tile)

        # Simple scaling to approx [0, 1] if min_energy <= 0
        map_energy_normalized = (map_energy - self.env_params.min_energy_per_tile) / energy_range
        map_energy_normalized[mask_map_energy] = -1.0 # Keep mask value distinct
        flat_parts.append(map_energy_normalized.flatten())

        # Map Features - Tile Type -> One-Hot
        tile_type = player_obs['map_features']['tile_type']

        # Ensure tile_type is within expected range [0, 2], handle -1 mask
        valid_tile_mask = (tile_type != -1)
        one_hot = np.zeros((W, H, 3), dtype=np.float32)

        # Only apply one-hot encoding where tile type is valid
        one_hot[valid_tile_mask & (tile_type == 0), 0] = 1.0 # Empty
        one_hot[valid_tile_mask & (tile_type == 1), 1] = 1.0 # Nebula
        one_hot[valid_tile_mask & (tile_type == 2), 2] = 1.0 # Asteroid

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
        max_possible_points = max(1, (self.env_params.max_steps_in_match + 1) * self.env_params.max_units)
        team_points = player_obs['team_points'].astype(np.float32)
        team_points = np.clip(team_points, 0, np.inf) / max_possible_points # Normalize by large heuristic value
        flat_parts.append(team_points.flatten())

        # Team Wins
        team_wins = player_obs['team_wins'].astype(np.float32) / max(1, self.env_params.match_count_per_episode)
        flat_parts.append(team_wins.flatten())

        # Steps (Global)
        total_max_steps = max(1, (self.env_params.max_steps_in_match + 1) * self.env_params.match_count_per_episode)
        steps = np.array([player_obs['steps']], dtype=np.float32) / total_max_steps
        flat_parts.append(steps.flatten()) # Should be shape (1,)

        # Match Steps
        match_steps = np.array([player_obs['match_steps']], dtype=np.float32) / max(1, self.env_params.max_steps_in_match)
        flat_parts.append(match_steps.flatten()) # Should be shape (1,)

        # Concatenate
        flat_obs = np.concatenate(flat_parts)
        return flat_obs.astype(np.float32)
