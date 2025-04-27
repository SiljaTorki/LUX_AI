from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

def get_base_model_config(random_seed: int, num_env_runners:int, num_envs_per_env_runner:int, matches_per_episode=505, num_gpus=0) -> PPOConfig:
    """
    Creates and returns PPO configuration object for the Lux AI environment.
    """
    train_batch_size = matches_per_episode * num_env_runners * num_envs_per_env_runner

    config = (
        PPOConfig()
        .environment(
            env="lux_base_multi_agent_env",
            env_config={"seed": random_seed}
        )
        .multi_agent(
            # policies={
            #     "shared_policy": PolicySpec(policy_class=None,)
            # },
            # policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
            # policies_to_train=["shared_policy"],
            policies={
                "p0": PolicySpec(policy_class=None),
                "p1": PolicySpec(policy_class=None)
            },
            policy_mapping_fn=(
                lambda agent_id, episode, **kwargs: "p0" if agent_id == "player_0" else "p1"
            ),
            policies_to_train=["p0", "p1"],
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
        )
        .rl_module(
            model_config={
                # LSTM config
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,    # How much history LSTM sees.
            }
        )
        .resources(
            num_gpus=num_gpus,
        )
        .evaluation(
                evaluation_interval=1,
                evaluation_num_workers=2,
        )
        .training(
            gamma=0.99,
            lr=0.00025,
            grad_clip = 0.5,
            #grad_clip_by = value,
            train_batch_size = train_batch_size,
            num_epochs = 5,
            minibatch_size = 512
        )
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
    )
    return config