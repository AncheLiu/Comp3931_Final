import _bootstrap  # noqa: F401
from experiment_runner import ValueExperimentConfig, parse_args, train_value_experiment


if __name__ == "__main__":
    args = parse_args()
    config = ValueExperimentConfig(env_name="LunarLander-v3", algorithm="dqn", episodes=500)
    train_value_experiment(config, args)
