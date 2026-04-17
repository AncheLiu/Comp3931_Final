import _bootstrap  # noqa: F401
from experiment_runner import ValueExperimentConfig, parse_args, train_value_experiment


if __name__ == "__main__":
    args = parse_args()
    config = ValueExperimentConfig(
        env_name="LunarLander-v3",
        algorithm="dueling_dqn_per_alpha04",
        episodes=1000,
        dueling_hidden_dim=128,
        use_per=True,
        per_alpha=0.4,
        loss_function="huber",
        grad_clip_norm=10.0,
    )
    train_value_experiment(config, args)
