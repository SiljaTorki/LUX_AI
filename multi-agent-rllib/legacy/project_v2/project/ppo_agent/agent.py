import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from .network import ActorCritic
from .buffer import RolloutBuffer
import os

class PPOAgent:
    def __init__(self, num_envs, obs_space_shape, action_space, config):
        """
        PPO Agent for Lux AI S3.

        Args:
            num_envs (int): Number of parallel environments.
            obs_space_shape (tuple): Shape of the single-player observation space (flattened).
            action_space (gym.spaces.Dict): The single-player action space.
            config (dict): Configuration dictionary. Must include keys like:
                           'num_steps_per_rollout', 'learning_rate', 'gamma', 'lambda_',
                           'num_epochs', 'batch_size', 'clip_coef', 'vf_coef',
                           'ent_coef', 'sap_coef', 'use_gpu', 'hidden_dim', etc.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu")
        self.action_space = action_space
        self.num_envs = num_envs
        self.num_steps_per_rollout = config['num_steps_per_rollout']

        num_inputs = np.prod(obs_space_shape)
        num_outputs_action_type = action_space["action_type"].nvec[0]
        num_outputs_sap_target = action_space["sap_target"].shape[1]
        self.max_units = action_space["action_type"].shape[0]

        self.network = ActorCritic(
            num_inputs,
            num_outputs_action_type * self.max_units,
            num_outputs_sap_target * self.max_units,
            hidden_dim=config.get("hidden_dim", 256)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.get("learning_rate", 3e-4),
            eps=config.get("adam_eps", 1e-5)
        )

        # Initialize the Rollout Buffer
        self.buffer = RolloutBuffer(
            self.num_steps_per_rollout,
            self.num_envs,
            obs_space_shape,
            action_space,
            self.device
        )

    def save_checkpoint(self, filepath):
        """ Saves the agent's network and optimizer state. """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """ Loads the agent's network and optimizer state. """
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint from {filepath}: {e}")
            return False

    def get_action_and_value(self, obs, deterministic=False):
        """ Gets actions and value estimates for a batch of observations. """
        if isinstance(obs, np.ndarray):
            # Assume obs comes from vectorized env, shape (num_envs, *obs_shape)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_type_logits_flat, sap_target_coords_flat, values = self.network(obs)

            # Reshape outputs: (num_envs, max_units, num_dims)
            action_type_logits = action_type_logits_flat.view(self.num_envs, self.max_units, -1)
            sap_target_coords = sap_target_coords_flat.view(self.num_envs, self.max_units, -1)

            # Action Type Sampling - Flatten for distribution: (num_envs * max_units, num_action_types)
            dist = Categorical(logits=action_type_logits.view(-1, action_type_logits.size(-1)))

            if deterministic:
                action_types_flat = dist.mode
            else:
                action_types_flat = dist.sample()

            log_probs_flat = dist.log_prob(action_types_flat)

            # Reshape back to (num_envs, max_units)
            action_types = action_types_flat.view(self.num_envs, self.max_units)
            log_probs = log_probs_flat.view(self.num_envs, self.max_units)

        # Prepare actions dict for env.step(), convert back to numpy on CPU
        actions_dict = {
            'action_type': action_types.cpu().numpy().astype(np.int16),
            'sap_target': sap_target_coords.cpu().numpy().astype(np.int16) # Env expects int16
        }

        return actions_dict, action_types, sap_target_coords, log_probs, values


    def update(self):
        """
        Performs PPO update steps over the collected rollout data.
        Assumes buffer is filled and compute_returns_and_advantages has been called.
        """
        clip_coef = self.config.get('clip_coef', 0.2)
        vf_coef = self.config.get('vf_coef', 0.5)
        ent_coef = self.config.get('ent_coef', 0.01)
        sap_coef = self.config.get('sap_coef', 0.5)
        num_epochs = self.config.get('num_epochs', 10)
        batch_size = self.config.get('batch_size', 64)
        max_grad_norm = self.config.get('max_grad_norm', 0.5)


        for epoch in range(num_epochs):
            for batch in self.buffer.get(batch_size):
                obs_batch = batch["observations"]
                act_types_batch = batch["action_types"]
                sap_targets_batch = batch["sap_targets"]
                old_log_probs_batch = batch["log_probs"]
                advantages_batch = batch["advantages"]
                returns_batch = batch["returns"]
                old_values_batch = batch["values"]


                # Flatten actions and sap targets for network evaluation
                act_types_batch_flat = act_types_batch.view(-1) # Shape: (batch_size * max_units,)
                sap_targets_batch_flat = sap_targets_batch.view(-1, 2) # Shape: (batch_size * max_units, 2)

                new_log_probs_flat, entropy_flat, new_values, current_sap_targets_flat = self.network.evaluate_actions(
                    obs_batch,
                    act_types_batch_flat,
                    sap_targets_batch_flat # This argument isn't strictly needed by evaluate_actions currently, but pass for consistency
                )

                new_log_probs_batch = new_log_probs_flat.view(batch_size, self.max_units).sum(dim=1)
                old_log_probs_summed = old_log_probs_batch.sum(dim=1)
                entropy_batch = entropy_flat.view(batch_size, self.max_units).mean(dim=1)


                # Calculate Losses
                # Normalize advantages
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # Policy Loss (Ratio and Clipped Surrogate Objective)
                logratio = new_log_probs_batch - old_log_probs_summed
                ratio = torch.exp(logratio)
                pg_loss1 = -advantages_batch * ratio
                pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                # ALternatively: Value clipping (as in original PPO paper)
                # v_clipped = old_values_batch + torch.clamp(new_values - old_values_batch, -clip_coef, clip_coef)
                # v_loss_clipped = F.mse_loss(v_clipped, returns_batch, reduction='none')
                # v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                new_values = new_values.view(-1)
                v_loss_unclipped = F.mse_loss(new_values, returns_batch, reduction='none')
                v_loss = 0.5 * v_loss_unclipped.mean() # Simpler version without value clipping

                # Entropy Loss
                entropy_loss = entropy_batch.mean()

                # Conditional Sap Target Loss (MSE for sap actions only)
                sap_action_code = 5 # The integer code for the sap action
                sap_mask = (act_types_batch_flat == sap_action_code).float()

                # Calculate MSE loss between predicted and actual sap targets
                # current_sap_targets_flat: (batch_size * max_units, 2) - Network's current prediction
                # sap_targets_batch_flat: (batch_size * max_units, 2) - Actual targets stored in buffer
                sap_target_loss_unmasked = F.mse_loss(current_sap_targets_flat, sap_targets_batch_flat, reduction='none').mean(dim=1) # Mean over dx, dy -> Shape: (batch_size * max_units,)

                # Apply mask and calculate mean over *only* the sap actions that occurred
                sap_target_loss = (sap_target_loss_unmasked * sap_mask).sum() / (sap_mask.sum() + 1e-8) # Avoid division by zero


                # Total Loss
                loss = (pg_loss
                        - ent_coef * entropy_loss
                        + vf_coef * v_loss
                        + sap_coef * sap_target_loss)

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

        # Reset buffer after updates for this rollout are done
        self.buffer.reset()

        # TODO: Return loss metrics for logging if needed
        # return pg_loss.item(), v_loss.item(), entropy_loss.item(), sap_target_loss.item()

    # TODO: Add methods for saving/loading agent state (network and optimizer)