import _bootstrap  # noqa: F401
from experiment_runner import ValueExperimentConfig, parse_args, train_value_experiment


if __name__ == "__main__":
    args = parse_args()
    config = ValueExperimentConfig(
        env_name="MountainCar-v0",
        algorithm="dueling_dqn",
        episodes=200,
        reward_shaping=True,
    )
    train_value_experiment(config, args)
