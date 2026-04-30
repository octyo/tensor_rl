class TabularQAgent:
    def __init__(self, state_space_size, action_space_size, lr=0.1, gamma=0.99):
        # Table format
        self.q_table = np.zeros((state_space_size, action_space_size))
        pass

class TensorizedTabularQAgent:
    def __init__(self):
        # Tensor representation framework (CP/Tucker/TT applied directly to Q-table tensor)
        pass
