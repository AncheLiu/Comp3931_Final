import _bootstrap  # noqa: F401
from experiment_runner import PPOExperimentConfig, parse_args, train_ppo_experiment


if __name__ == "__main__":
    args = parse_args()
    config = PPOExperimentConfig(
        env_name="LunarLander-v3",
        algorithm="ppo_improved",
        episodes=3000,
        lr_actor=1e-4,
        lr_critic=3e-4,
        k_epochs=8,
        hidden_dim=128,
        entropy_coef=0.001,
        use_gae=True,
        gae_lambda=0.95,
        normalize_advantages=True,
        minibatch_size=64,
        max_grad_norm=0.5,
    )
    train_ppo_experiment(config, args)
