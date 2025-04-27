import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict, MultiDiscrete

class FlattenActionWrapper(gym.ActionWrapper):
    """
    Wrapper to flatten the Lux AI S3 dictionary action space into a single Box space
    suitable for SB3, assuming a single agent controls both players.

    The original action space is Dict({'player_0': p_space, 'player_1': p_space}),
    where p_space is Dict({'action_type': MultiDiscrete(N,), 'sap_target': Box(N, 2)}).

    This wrapper converts agent actions from a flat Box space back into the
    nested dictionary structure expected by the underlying env.
    """
    def __init__(self, env: gym.Env, multi_agent: bool = False):
        super().__init__(env)
        self.multi_agent = multi_agent
        self.env_params= env.env_params

        # --- Validation (slightly adjusted for flexibility) ---
        if not isinstance(env.action_space, GymDict):
            raise ValueError("This wrapper requires the underlying env's action_space to be a gymnasium.spaces.Dict.")
        if 'player_0' not in env.action_space.spaces:
            raise ValueError("Underlying action_space must contain at least 'player_0' key.")
        if not isinstance(env.action_space['player_0'], GymDict):
            raise ValueError("Player action space for player_0 must be a Dict.")
        if 'action_type' not in env.action_space['player_0'].spaces or \
                'sap_target' not in env.action_space['player_0'].spaces:
            raise ValueError("Player_0 action space Dict must contain 'action_type' and 'sap_target'.")
        # Check player_1 only if it exists (might wrap an env already processed for single player)
        if 'player_1' in env.action_space.spaces:
            if not isinstance(env.action_space['player_1'], GymDict):
                raise ValueError("Player action space for player_1 must be a Dict.")
            if env.action_space['player_0'].spaces.keys() != env.action_space['player_1'].spaces.keys():
                raise ValueError("Player 0 and Player 1 action spaces must have the same structure.")
        elif self.multi_agent:
            # If multi_agent=True, we expect player_1 to exist for flattening.
            raise ValueError("Multi-agent mode requires 'player_1' in the underlying action space.")

        self.original_player_action_space = env.action_space['player_0']
        # Ensure nvec exists and has expected structure before accessing shape
        action_type_space = self.original_player_action_space['action_type']
        if not isinstance(action_type_space, MultiDiscrete) or not hasattr(action_type_space, 'nvec'):
            raise ValueError("action_type space must be MultiDiscrete with nvec attribute.")
        self.max_units = action_type_space.nvec.shape[0]

        # Calculate size for a *single* player's flattened action
        # N action types + N*2 sap target coords = 3*N
        self.single_player_flat_size = self.max_units * 3

        # Define the flattened Box space for one player
        single_player_flat_action_space = Box(
            low=-1.0, high=1.0, shape=(self.single_player_flat_size,), dtype=np.float32
        )

        # --- Define the final action_space based on mode ---
        if self.multi_agent:
            # Multi-agent: Dict of flat Boxes per player
            self.action_space = GymDict({
                'player_0': single_player_flat_action_space,
                'player_1': single_player_flat_action_space,
            })
        else:
            # Single-agent: Concatenated flat Box for both players
            # N(p0 type) + N*2(p0 target) + N(p1 type) + N*2(p1 target) = 6*N
            total_flat_size = self.single_player_flat_size * 2
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(total_flat_size,), dtype=np.float32
            )
        # --- End action_space definition ---

        # Store original bounds for sap target un-normalization
        sap_space = self.original_player_action_space['sap_target']
        if not isinstance(sap_space, Box) or sap_space.shape != (self.max_units, 2):
            raise ValueError("sap_target space must be a Box with shape (max_units, 2).")
        self.sap_low = sap_space.low[0, 0] # Assuming uniform bounds
        self.sap_high = sap_space.high[0, 0]
        self.sap_range = self.sap_high - self.sap_low
        if self.sap_range <= 0:
            # Handle zero range case, maybe default to range of 1 or log warning
            self.sap_range = 1.0


    def _unnormalize_action_type(self, norm_action_type):
        """ Maps [-1, 1] back to discrete 0-5 """
        # Scale from [-1, 1] to [0, 5]
        # (val + 1) / 2 maps [-1, 1] to [0, 1]
        # Multiply by 5 to get [0, 5]
        # Round and clip
        unnorm = np.round(((norm_action_type + 1.0) / 2.0) * 5.0)
        return np.clip(unnorm, 0, 5).astype(np.int16)

    def _unnormalize_sap_target(self, norm_sap_target):
        """ Maps [-1, 1] back to original sap target range [low, high] """
        # Scale from [-1, 1] to [0, 1]
        zero_one = (norm_sap_target + 1.0) / 20.
        # Scale to [0, range] and add low
        unnorm = (zero_one * self.sap_range) + self.sap_low
        # Round to nearest integer as env expects int16
        return np.round(unnorm).astype(np.int16)

    def _decode_player_action(self, flat_action):
        """ Decodes a single player's flat action vector. """
        if flat_action.shape != (self.single_player_flat_size,):
            raise ValueError(f"Incorrect single player flat action shape. Expected {(self.single_player_flat_size,)}, got {flat_action.shape}")

        N = self.max_units
        action_type_norm = flat_action[0:N]
        sap_target_flat_norm = flat_action[N : N + N*2] # Shape (N*2,)

        action_type = self._unnormalize_action_type(action_type_norm)
        sap_target = self._unnormalize_sap_target(sap_target_flat_norm.reshape(N, 2))

        return {"action_type": action_type, "sap_target": sap_target}

    def action(self, action) -> dict:
        """
        Converts the flattened action(s) back into the dictionary format.
        Args:
            action:
                If multi_agent=True: A dict {'player_0': flat_p0, 'player_1': flat_p1}
                If multi_agent=False: A numpy array shape (6 * max_units,).
        Returns:
            A dictionary structured for the underlying env's step method.
        """
        original_format_action = {}

        if self.multi_agent:
            if not isinstance(action, dict) or 'player_0' not in action or 'player_1' not in action:
                raise ValueError("In multi_agent mode, action must be a dict with 'player_0' and 'player_1' keys.")
            original_format_action['player_0'] = self._decode_player_action(action['player_0'])
            original_format_action['player_1'] = self._decode_player_action(action['player_1'])
        else:
            # Single-agent mode: input is a single flat array for both players
            if not isinstance(action, np.ndarray) or action.shape != self.action_space.shape:
                raise ValueError(f"Incorrect single-agent action shape. Expected {self.action_space.shape}, got {action.shape if isinstance(action, np.ndarray) else type(action)}")

            N = self.max_units
            # Split the combined flat action array (total size 6N)
            p0_flat_action = action[0 : self.single_player_flat_size]          # Indices 0 to 3N-1
            p1_flat_action = action[self.single_player_flat_size : ] # Indices 3N to 6N-1

            original_format_action['player_0'] = self._decode_player_action(p0_flat_action)
            original_format_action['player_1'] = self._decode_player_action(p1_flat_action)

        return original_format_action