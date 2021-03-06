import torch.nn as nn
import torch.nn.functional as F
import torch
import os
# ====================================== helper functions ======================================
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from common import make_grid_map, get_surrounding, get_observations


# ====================================== define algo ===========================================
# todo
class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# todo
class DQN(object):
    def __init__(self):
        # pass
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_size = 64

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        # pass
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()

        return action

    def load(self, file):
        # pass
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file))


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
# todo
agent = DQN()
agent.load('critic_25000.pth')

# ================================================================================================
"""
input:
    observation: dict
    {
        1: ?????????
        2: ????????????????????????
        3???????????????????????????
        "board_width": ???????????????
        "board_height"??????????????????
        "last_direction"?????????????????????????????????
        "controlled_snake_index"????????????????????????????????????2???3???
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""


# todo
def my_controller(observation, action_space_list, is_act_continuous):
    # pass
    obs = get_observations(observation, 0, 18)
    action = agent.choose_action(obs)
    action_ = to_joint_action(action, 1)
    return action_
