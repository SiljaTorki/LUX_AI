import torch
import numpy as np

def compute_gae(rewards, values, dones, next_value, gamma, lambda_):
    """
    Computes Generalized Advantage Estimation (GAE).

    Args:
        rewards (torch.Tensor): Tensor of rewards collected during rollouts. Shape: (num_steps, num_envs).
        values (torch.Tensor): Tensor of value estimates for each state. Shape: (num_steps, num_envs).
        dones (torch.Tensor): Tensor indicating whether a step was terminal. Shape: (num_steps, num_envs).
        next_value (torch.Tensor): Value estimate for the state after the last rollout step. Shape: (num_envs,).
        gamma (float): Discount factor.
        lambda_ (float): GAE smoothing parameter.

    Returns:
        torch.Tensor: Computed advantages. Shape: (num_steps, num_envs).
        torch.Tensor: Computed returns (advantages + values). Shape: (num_steps, num_envs).
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0

    # Ensure values includes the next_value for calculation convenience
    full_values = torch.cat((values, next_value.unsqueeze(0)), dim=0) # Shape: (num_steps + 1, num_envs)

    for t in reversed(range(num_steps)):
        # If the episode ended at step t, the value of the next state is 0
        # mask is 0 if done, 1 if not done
        mask = 1.0 - dones[t].float()

        # Calculate delta: delta_t = r_t + gamma * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[t] + gamma * full_values[t + 1] * mask - full_values[t]

        # Calculate advantage: A_t = delta_t + gamma * lambda * A_{t+1} * mask
        advantages[t] = last_gae_lam = delta + gamma * lambda_ * last_gae_lam * mask

    # Calculate returns: R_t = A_t + V(s_t)
    returns = advantages + values

    return advantages, returns