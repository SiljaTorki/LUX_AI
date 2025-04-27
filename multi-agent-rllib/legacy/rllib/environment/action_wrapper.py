import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class FlattenActionWrapper(gym.ActionWrapper):
    """
    Wrapper to flatten the Lux AI S3 dictionary action space into a single Box space
    suitable for SB3, assuming a single agent controls both players.

    The original action space is Dict({'player_0': p_space, 'player_1': p_space}),
    where p_space is Dict({'action_type': MultiDiscrete(N,), 'sap_target': Box(N, 2)}).

    This wrapper converts agent actions from a flat Box space back into the
    nested dictionary structure expected by the underlying env.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.action_space, Dict):
            raise ValueError("This wrapper requires the underlying env's action_space to be a gymnasium.spaces.Dict.")
        if 'player_0' not in env.action_space.spaces or 'player_1' not in env.action_space.spaces:
            raise ValueError("Underlying action_space must contain 'player_0' and 'player_1' keys.")
        if not isinstance(env.action_space['player_0'], Dict) or \
                not isinstance(env.action_space['player_1'], Dict):
            raise ValueError("Player action spaces must be Dicts.")
        if 'action_type' not in env.action_space['player_0'].spaces or \
                'sap_target' not in env.action_space['player_0'].spaces:
            raise ValueError("Player action space Dict must contain 'action_type' and 'sap_target'.")

        self.original_player_action_space = env.action_space['player_0'] # Assumes player_1 is identical
        self.max_units = self.original_player_action_space['action_type'].nvec.shape[0]

        # Calculate total size: N (p0 type) + N*2 (p0 target) + N (p1 type) + N*2 (p1 target) = 6*N
        flat_action_size = self.max_units * 6

        # Define bounds for the Box space. We'll normalize actions to [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(flat_action_size,), dtype=np.float32)

        # Store original bounds for un-normalizing
        self.sap_low = self.original_player_action_space['sap_target'].low[0, 0] # Assuming uniform bounds
        self.sap_high = self.original_player_action_space['sap_target'].high[0, 0]
        self.sap_range = self.sap_high - self.sap_low

    def _unnormalize_action_type(self, norm_action_type):
        """ Maps [-1, 1] back to discrete 0-5 """
        # Scale from [-1, 1] to [0, 5]
        # (val + 1) / 2 maps [-1, 1] to [0, 1]
        # Multiply by 5 to get [0, 5]
        # Round and clip
        unnorm = np.round(((norm_action_type + 1) / 2) * 5)
        return np.clip(unnorm, 0, 5).astype(np.int16)

    def _unnormalize_sap_target(self, norm_sap_target):
        """ Maps [-1, 1] back to original sap target range [low, high] """
        # Scale from [-1, 1] to [0, 1]
        zero_one = (norm_sap_target + 1) / 2
        # Scale to [0, range] and add low
        unnorm = (zero_one * self.sap_range) + self.sap_low
        # Round to nearest integer as env expects int16
        return np.round(unnorm).astype(np.int16)

    def action(self, action: np.ndarray) -> dict:
        """
        Converts the flattened Box action back into the dictionary format.
        Args:
            action: A numpy array from the agent, shape (6 * max_units,).
        Returns:
            A dictionary structured for the underlying env's step method.
        """
        if action.shape != self.action_space.shape:
            raise ValueError(f"Incorrect action shape. Expected {self.action_space.shape}, got {action.shape}")

        N = self.max_units
        # Split the flat action array
        p0_action_type_norm = action[0:N]
        p0_sap_target_flat_norm = action[N : N + N*2]
        p1_action_type_norm = action[N + N*2 : N + N*2 + N]
        p1_sap_target_flat_norm = action[N + N*2 + N : N + N*2 + N + N*2]

        # Reshape and un-normalize
        p0_action_type = self._unnormalize_action_type(p0_action_type_norm)
        p0_sap_target = self._unnormalize_sap_target(p0_sap_target_flat_norm.reshape(N, 2))

        p1_action_type = self._unnormalize_action_type(p1_action_type_norm)
        p1_sap_target = self._unnormalize_sap_target(p1_sap_target_flat_norm.reshape(N, 2))

        # Construct the dictionary action
        original_format_action = {
            "player_0": {
                "action_type": p0_action_type,
                "sap_target": p0_sap_target
            },
            "player_1": {
                "action_type": p1_action_type,
                "sap_target": p1_sap_target
            }
        }
        return original_format_action