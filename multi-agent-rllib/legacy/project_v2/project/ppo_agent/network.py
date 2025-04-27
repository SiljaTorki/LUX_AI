import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs_action_type, num_outputs_sap_target, hidden_dim=256):
        """
        Combined Actor-Critic Network for Lux AI S3.

        Args:
            num_inputs (int): Dimension of the flattened and normalized observation space.
            num_outputs_action_type (int): Number of discrete action types (should be 6).
            num_outputs_sap_target (int): Number of outputs for sap target (should be 2 for dx, dy).
            hidden_dim (int): Size of the hidden layers.
        """
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: Action Type Logits
        self.action_type_head = nn.Linear(hidden_dim, num_outputs_action_type)

        # Actor head: Sap Target Coordinates (Direct Regression)
        self.sap_target_head = nn.Linear(hidden_dim, num_outputs_sap_target)

        # Critic head: Value Estimate
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor (flattened observation).

        Returns:
            tuple: (action_type_logits, sap_target_coords, value_estimate)
                   - action_type_logits (torch.Tensor): Logits for the action type distribution.
                   - sap_target_coords (torch.Tensor): Predicted dx, dy for sap action.
                   - value_estimate (torch.Tensor): Estimated state value.
        """
        shared_features = self.shared_layers(x)

        action_type_logits = self.action_type_head(shared_features)
        sap_target_coords = self.sap_target_head(shared_features) # Direct output
        value_estimate = self.value_head(shared_features)

        return action_type_logits, sap_target_coords, value_estimate


    def evaluate_actions(self, obs, action_types_flat, sap_targets_flat):
        """
        Evaluates the log probability and entropy of given actions,
        and returns the value estimate and predicted sap targets for the given observations.
        Designed for use during the PPO update step.

        Args:
            obs (torch.Tensor): Batch of observations. Shape (batch_size, num_inputs).
            action_types_flat (torch.Tensor): Batch of action types taken. Shape (batch_size * max_units,).
            sap_targets_flat (torch.Tensor): Batch of sap targets taken. Shape (batch_size * max_units, 2).

        Returns:
            tuple: (new_log_probs_flat, entropy_flat, values, current_sap_targets_flat)
                   - new_log_probs_flat (torch.Tensor): Log probability of the input action types under the current policy. Shape (batch_size * max_units,).
                   - entropy_flat (torch.Tensor): Entropy of the action type distribution. Shape (batch_size * max_units,).
                   - values (torch.Tensor): Value estimates for the observations. Shape (batch_size, 1).
                   - current_sap_targets_flat (torch.Tensor): Sap targets predicted by the *current* network. Shape (batch_size * max_units, 2).
        """
        action_type_logits_flat, current_sap_targets_flat, values = self.forward(obs)
        batch_size = obs.size(0)
        max_units = action_types_flat.size(0) // batch_size if batch_size > 0 else 0 # Handle empty batch case? No, PPO loop shouldn't have empty batches.
        num_action_types = action_type_logits_flat.size(-1) // max_units


        # Action Type Distribution - Reshape logits: (batch_size * max_units, num_action_types)
        dist = Categorical(logits=action_type_logits_flat.view(-1, num_action_types))

        new_log_probs_flat = dist.log_prob(action_types_flat) # Log prob of the actions *actually taken*
        entropy_flat = dist.entropy()

        # Reshape predicted sap targets to match the flat action structure
        current_sap_targets_flat = current_sap_targets_flat.view(-1, 2) # Shape: (batch_size * max_units, 2)

        return new_log_probs_flat, entropy_flat, values, current_sap_targets_flat

    def get_action_value(self, obs, action_type=None, sap_target=None):
        """
        Computes actions, log probabilities, entropy, and value for a given observation.

        Args:
            obs (torch.Tensor): Input observation.
            action_type (torch.Tensor, optional): Specific action type to evaluate log_prob for. Defaults to None (sample).
            sap_target (torch.Tensor, optional): Specific sap target to evaluate log_prob for (NOT IMPLEMENTED YET FOR REGRESSION). Defaults to None (sample).

        Returns:
            tuple: (sampled_action_type, sampled_sap_target, log_prob, entropy, value) OR
                   (log_prob, entropy, value) if action is provided.
        """
        action_type_logits, sap_target_coords, value = self.forward(obs)

        # Action Type Distribution
        action_probs = Categorical(logits=action_type_logits)

        if action_type is None:
            # Sample action type if not provided
            action_type = action_probs.sample()

        # Calculate log probability and entropy for the action type
        action_type_log_prob = action_probs.log_prob(action_type)
        entropy = action_probs.entropy()

        if action_type is None: # If we were sampling
            return action_type, sap_target_coords, action_type_log_prob, entropy, value
        else: # If we were evaluating provided actions
            return action_type_log_prob, entropy, value


    def get_value(self, obs):
        """
        Computes only the value estimate for a given observation.

        Args:
            obs (torch.Tensor): Input observation.

        Returns:
            torch.Tensor: Estimated state value.
        """
        _, _, value = self.forward(obs)
        return value