import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple fully connected Q-network used by DQN-style agents."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    brain = QNetwork(state_dim=4, action_dim=2)
    dummy_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    q_values = brain(dummy_state)
    best_action = torch.argmax(q_values).item()
    print(f"Q values: {q_values}")
    print(f"Best action index: {best_action}")
