import _bootstrap  # noqa: F401
from experiment_runner import PPOExperimentConfig, parse_args, train_ppo_experiment


if __name__ == "__main__":
    args = parse_args()
    config = PPOExperimentConfig(
        env_name="MountainCar-v0",
        algorithm="ppo",
        episodes=3000,
        reward_shaping=True,
    )
    train_ppo_experiment(config, args)
