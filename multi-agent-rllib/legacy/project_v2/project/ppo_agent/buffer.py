import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_space, device):
        """
        Rollout Buffer for PPO. Stores trajectories for multiple environments.

        Args:
            num_steps (int): Number of steps to collect per env before calculating advantages.
            num_envs (int): Number of parallel environments.
            obs_shape (tuple): Shape of a single observation.
            action_space (gym.spaces.Dict): Action space of the env (for shapes).
            device (torch.device): CPU or CUDA device.
        """
        self.device = device
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        # Extract shapes from the action space Dict
        self.action_type_shape = action_space["action_type"].shape
        self.sap_target_shape = action_space["sap_target"].shape

        # Initialize storage tensors on the specified device
        self.observations = torch.zeros((self.num_steps, self.num_envs) + self.obs_shape, dtype=torch.float32).to(device)
        # Store both action components
        self.action_types = torch.zeros((self.num_steps, self.num_envs) + self.action_type_shape, dtype=torch.int64).to(device)
        self.sap_targets = torch.zeros((self.num_steps, self.num_envs) + self.sap_target_shape, dtype=torch.float32).to(device) # Assuming float for regression targets
        self.log_probs = torch.zeros((self.num_steps, self.num_envs) + self.action_type_shape, dtype=torch.float32).to(device) # Log prob usually corresponds to sampled discrete action
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(device) # Use float for mask calculations
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(device)

        # Placeholders for calculated advantages and returns
        self.advantages = torch.zeros_like(self.rewards).to(device)
        self.returns = torch.zeros_like(self.rewards).to(device)

        self.step = 0 # Current step counter within the buffer
        self.buffer_filled = False


    def add(self, obs, action_type, sap_target, reward, done, value, log_prob):
        """ Adds a transition from each env to the buffer. """
        if self.step >= self.num_steps:
            raise ValueError("Buffer is full.")

        # Store directly on device
        self.observations[self.step] = obs
        self.action_types[self.step] = action_type
        self.sap_targets[self.step] = sap_target
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float() # Store dones as float for masking
        self.values[self.step] = value.flatten()
        self.log_probs[self.step] = log_prob

        self.step += 1
        if self.step == self.num_steps:
            self.buffer_filled = True

    def compute_returns_and_advantages(self, last_value, gamma, lambda_):
        """
        Computes returns and advantages (GAE) for the collected rollout.

        Args:
            last_value (torch.Tensor): Value estimate for the state after the last step. Shape (num_envs,).
            gamma (float): Discount factor.
            lambda_ (float): GAE parameter.
        """
        if not self.buffer_filled:
            print(f"Warning: Computing returns on partially filled buffer (step {self.step}/{self.num_steps})")
            # Handle partial buffer if necessary, maybe only compute for filled part?
            # assume for now that we only call this when full or at episode end.

        from .utils import compute_gae # Avoid circular import if utils imports buffer
        self.advantages, self.returns = compute_gae(
            self.rewards[:self.step].to(self.device),
            self.values[:self.step].to(self.device),
            self.dones[:self.step].to(self.device),
            last_value.to(self.device).flatten(),
            gamma,
            lambda_
        )

    def get(self, batch_size):
        """
        Generates batches of experience data for training.

        Args:
            batch_size (int): The size of each mini-batch.

        Yields:
            dict: A dictionary containing tensors for a mini-batch of experience.
                  Keys: "observations", "action_types", "sap_targets", "values",
                        "log_probs", "advantages", "returns".
        """
        if not self.buffer_filled:
            raise ValueError("Buffer not filled. Cannot generate batches.")
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages and returns not computed. Call compute_returns_and_advantages first.")

        # Flatten the data across environments and steps
        num_samples = self.num_steps * self.num_envs
        indices = np.random.permutation(num_samples)

        # Flatten the data (assuming it's already on self.device)
        flat_obs = self.observations[:self.step].reshape(num_samples, *self.obs_shape)
        flat_action_types = self.action_types[:self.step].reshape(num_samples, *self.action_type_shape)
        flat_sap_targets = self.sap_targets[:self.step].reshape(num_samples, *self.sap_target_shape)
        flat_log_probs = self.log_probs[:self.step].reshape(num_samples, *self.action_type_shape)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values[:self.step].reshape(-1)

        start_idx = 0
        while start_idx < num_samples:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield {
                "observations": flat_obs[batch_indices],
                "action_types": flat_action_types[batch_indices],
                "sap_targets": flat_sap_targets[batch_indices],
                "values": flat_values[batch_indices],
                "log_probs": flat_log_probs[batch_indices], # Old log_probs# Old log_probs
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
            }
            start_idx += batch_size

    def reset(self):
        """Resets the buffer step counter and filled status."""
        self.step = 0
        self.buffer_filled = False
        # Maybe we should clear tensors if memory is a concern, but overwriting should be fine