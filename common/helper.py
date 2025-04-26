import numpy as np
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback

def make_env(env_params, wrapped_env, seed=None):
    """
    Create and wrap a Lux S3 environment for SBX/Stable Baselines 3.

    Args:
        seed: Random seed for reproducibility
        player_id: ID of the player to train ('player_0' or 'player_1')
        opponent_strategy: Strategy for the opponent ('random', 'static', etc.)
    """

    # Reset to initialize observation and action spaces
    seed_val = seed if seed is not None else np.random.randint(10000)
    wrapped_env.reset(seed=seed_val, options=dict(params=env_params))

    return wrapped_env


def custom_env_check(env, episodes=1):
    for episode in range(episodes):
        env.reset()
        done = False
        rewards = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            rewards += reward
            env.render()
            if done or truncated:
                done = True
        print("🏁 Episode finished 🏁", rewards)
    env.close()
    
def linear_schedule(
    initial_value: float, final_value: float
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate
    :param final_value: Final learning rate
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: 1.0 - (current_timestep / total_timesteps)
        :return: current learning rate
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return func

class EnhancedTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnhancedTensorboardCallback, self).__init__(verbose)
        self.match_rewards = [
            [] for _ in range(5)
        ]  # For tracking rewards across 5 matches
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            metrics = info["lux_metrics"]
            metrics_static_planner = info["lux_metrics_static_planner"] if "lux_metrics_static_planner" in info else {}
            metrics_mappo = info["lux_metrics_mappo"] if "lux_metrics_mappo" in info else {}
            # if "episode" in info:
            #     # Basic episode metrics
            #     self.episode_rewards.append(info["episode"]["r"])
            #     self.episode_lengths.append(info["episode"]["l"])
            #     self.episode_count += 1

            #     # Log to TensorBoard
            #     self.logger.record(
            #         "rollout/ep_rew_mean",
            #         sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:]),
            #     )
            #     self.logger.record(
            #         "rollout/ep_len_mean",
            #         sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:]),
            #     )

            #     # You can log individual episode rewards too
            #     self.logger.record(f"episode/reward", info["episode"]["r"])
            #     self.logger.record(f"episode/length", info["episode"]["l"])
            #     if "match_number" in metrics and "game_number" in metrics:
            #         match_num = metrics["match_number"]
            #         game_num = metrics["game_number"]
            #         self.logger.record(
            #             f"lux/game_{game_num}_match_{match_num}_reward",
            #             info["episode"]["r"],
            #         )

            #         # Track reward progression across matches
            #         if 0 <= match_num < 5:
            #             self.match_rewards[match_num].append(info["episode"]["r"])
            #             match_avg = sum(self.match_rewards[match_num]) / len(
            #                 self.match_rewards[match_num]
            #             )
            #             self.logger.record(
            #                 f"lux/match_{match_num}_avg_reward", match_avg
                        # )

            # print(f"Match {metrics['match_number']}, Game {metrics['game_number']}, Reward: {info['episode']['r']}")
            # Resource metrics
            if "energy_collected" in metrics:
                value = metrics["energy_collected"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/energy_collected", float_value)
            if "total_energy" in metrics:
                value = metrics["total_energy"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/total_energy", float_value)
            # Exploration metrics
            if "map_coverage" in metrics:
                value = metrics["map_coverage"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/map_coverage", float_value)
            if "new_tiles_revealed" in metrics:
                value = metrics["new_tiles_revealed"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/new_tiles_revealed", float_value)

            if "points_earned" in metrics:
                value = metrics["points_earned"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/points_earned", float_value)
            if "rule_reward" in metrics:
                value = metrics["rule_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/rule_reward", float_value)
            if "sap_reward" in metrics:
                value = metrics["sap_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/sap_reward", float_value)
            if "sap_actions_taken" in metrics:
                value = metrics["sap_actions_taken"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/sap_actions_taken", float_value)
            if "point_reward" in metrics:
                value = metrics["point_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/point_reward", float_value)
            if "final_reward" in metrics:
                value = metrics["final_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/final_reward", float_value)

            if "relic_control_streak" in metrics:
                value = metrics["relic_control_streak"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/relic_control_streak", float_value)
            if "visited_tiles_count" in metrics:
                value = metrics["visited_tiles_count"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/visited_tiles_count", float_value)
            if "relic_point_tiles_found" in metrics:
                value = metrics["relic_point_tiles_found"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/relic_point_tiles_found", float_value)
            # Win metrics
            if "win_rate" in metrics:
                value = metrics["win_rate"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/win_rate", float_value)

            if "unit_positions" in metrics:
                unit_positions = metrics["unit_positions"]
                # Log each unit's position
                for idx, pos in enumerate(unit_positions):
                    self.logger.record(f"lux/unit_{idx}_position", pos)
            if "static_planner_bonus_sap_reward" in metrics_static_planner:
                value = metrics_static_planner["static_planner_bonus_sap_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record(
                    "lux/static_planner_bonus_sap_reward", float_value
                )
            if "mappo_reward" in metrics_mappo:
                value = metrics_mappo["mappo_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/mappo_reward", float_value)
            if "final_mappo_reward" in metrics_mappo:
                value = metrics_mappo["final_mappo_reward"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/final_mappo_reward", float_value)
            if "collision_detected" in metrics_mappo:
                value = metrics_mappo["collision_detected"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/collision_detected", float_value)
            if "replan_decisions" in metrics_mappo:
                value = metrics_mappo["replan_decisions"]
                # Convert JAX array to standard float
                if hasattr(value, "item"):
                    float_value = float(value.item())
                else:
                    float_value = float(value)
                self.logger.record("lux/replan_decisions", float_value)
        return True
