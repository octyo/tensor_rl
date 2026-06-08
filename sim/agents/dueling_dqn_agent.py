from agents.double_dqn_agent import DoubleDQNAgent
from models.q_networks import DuelingQNetwork


class DuelingDQNAgent(DoubleDQNAgent):
    """Dueling DQN = Double DQN update rule + dueling V/A network architecture.
    The only difference from DoubleDQNAgent is that it expects a DuelingQNetwork.
    Agent logic (select_action, update, soft target update) is unchanged.
    """
    pass


def make_dueling_agent(state_dim: int, action_dim: int, network_type: str, rank: int,
                       hidden_sizes=(128, 128), **kwargs) -> DuelingDQNAgent:
    q_net = DuelingQNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes,
                            network_type=network_type, rank=rank)
    return DuelingDQNAgent(q_net, action_dim=action_dim, **kwargs)
