import _bootstrap  # noqa: F401
from experiment_runner import ValueExperimentConfig, parse_args, train_value_experiment


if __name__ == "__main__":
    args = parse_args()
    config = ValueExperimentConfig(
        env_name="MountainCar-v0",
        algorithm="dueling_dqn_per",
        episodes=700,
        reward_shaping=True,
        dueling_hidden_dim=128,
        epsilon_decay=0.99,
        use_per=True,
        loss_function="huber",
        grad_clip_norm=10.0,
    )
    train_value_experiment(config, args)
