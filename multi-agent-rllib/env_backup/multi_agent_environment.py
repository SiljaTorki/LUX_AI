import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .observation_wrapper import FlattenNormalizeObservation
from .action_wrapper import FlattenActionWrapper
from .environment import LuxEnvBase

class LuxMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self._base_env = LuxEnvBase(random_seed=config.get("seed", 42))
        self._obs_wrapped_env = FlattenNormalizeObservation(env=self._base_env, multi_agent=True)
        self._action_wrapped_env = FlattenActionWrapper(env=self._obs_wrapped_env, multi_agent=True)
        self._env = self._action_wrapped_env
        self.env_params = self._env.env_params

        self.agents = self.possible_agents =["player_0", "player_1"]
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._skip_agent_checking = True

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        return obs_dict, info_dict

    def step(self, action_dict):
        # Assuming self._env.step now returns (obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict)
        # The variable names dones_dict and truncs_dict should be updated to terminated_dict and truncated_dict
        obs_dict, rewards_dict, terminated_dict, truncated_dict, raw_infos_dict = self._env.step(action_dict)

        filtered_infos_dict = {}
        for agent_id in self.agents: # self.agents = ["player_0", "player_1"]
            if agent_id in raw_infos_dict:
                filtered_infos_dict[agent_id] = raw_infos_dict[agent_id]
            else:
                # Handle cases where an agent might not have info (e.g., if done)
                # RLlib usually expects an entry for active agents.
                filtered_infos_dict[agent_id] = {}

        # The base env step should now return dicts with __all__ already correctly set.
        # No need to recalculate here, just ensure they are passed through.
        # If the base env doesn't set __all__, you would calculate it here:
        if "__all__" not in terminated_dict:
            terminated_dict["__all__"] = terminated_dict.get("player_0", False) or terminated_dict.get("player_1", False)
        if "__all__" not in truncated_dict:
            truncated_dict["__all__"] = truncated_dict.get("player_0", False) or truncated_dict.get("player_1", False)

        # Rename dones -> terminated, truncs -> truncated for clarity matching Gymnasium
        return obs_dict, rewards_dict, terminated_dict, truncated_dict, filtered_infos_dict

    def close(self):
        self._env.close()