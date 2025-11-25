import numpy as np

class UniformPolicy:
    def __init__(self, action_state_size) -> None:
        self.action_state_size = action_state_size

    def get_move(self, state):
        return np.random.randint(0, self.action_state_size)
