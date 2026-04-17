import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer used by the baseline DQN-style algorithms."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return _format_batch(batch)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay using a sum tree for efficient sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = [None] * capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self._update_tree(self.position, self.max_priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        total_priority = float(self.tree[0])
        if total_priority <= 0:
            valid_indices = random.choices(range(self.size), k=batch_size)
            batch = [self.buffer[index] for index in valid_indices]
            state, action, reward, next_state, done = _format_batch(batch)
            weights = np.ones(batch_size, dtype=np.float32)
            return state, action, reward, next_state, done, valid_indices, weights

        segment = total_priority / batch_size
        indices = []
        probabilities = []
        batch = []

        for batch_index in range(batch_size):
            lower = segment * batch_index
            upper = segment * (batch_index + 1)
            sample_value = min(random.uniform(lower, upper), np.nextafter(total_priority, 0.0))
            tree_index = self._retrieve(0, sample_value)
            data_index = tree_index - self.capacity + 1
            priority = self.tree[tree_index]

            if data_index >= self.size or self.buffer[data_index] is None or priority <= 0:
                data_index = random.randrange(self.size)
                priority = max(self.tree[data_index + self.capacity - 1], self.epsilon)

            indices.append(data_index)
            probabilities.append(priority / total_priority)
            batch.append(self.buffer[data_index])

        probabilities = np.array(probabilities, dtype=np.float32)
        probabilities = np.maximum(probabilities, self.epsilon)
        weights = (self.size * probabilities) ** (-beta)
        weights = weights / weights.max()

        state, action, reward, next_state, done = _format_batch(batch)
        return state, action, reward, next_state, done, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            new_priority = float(abs(priority)) + self.epsilon
            self.max_priority = max(self.max_priority, new_priority)
            self._update_tree(index, new_priority)

    def __len__(self):
        return self.size

    def _update_tree(self, data_index: int, priority: float):
        tree_index = data_index + self.capacity - 1
        scaled_priority = priority ** self.alpha
        change = scaled_priority - self.tree[tree_index]
        self.tree[tree_index] = scaled_priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def _retrieve(self, tree_index: int, value: float) -> int:
        left = 2 * tree_index + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_index

        if self.tree[left] > 0 and value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])


def _format_batch(batch):
    state = np.array([item[0] for item in batch], dtype=np.float32)
    action = np.array([item[1] for item in batch], dtype=np.int64)
    reward = np.array([item[2] for item in batch], dtype=np.float32)
    next_state = np.array([item[3] for item in batch], dtype=np.float32)
    done = np.array([item[4] for item in batch], dtype=np.float32)
    return state, action, reward, next_state, done


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)
    dummy_state = np.array([0.1, 0.2, 0.3, 0.4])
    buffer.push(dummy_state, 1, 1.0, dummy_state, False)
    print(f"Replay buffer size: {len(buffer)}")
