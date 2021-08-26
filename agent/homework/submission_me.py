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
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_size = 64
        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        inference_output = self.inference(observation)
        return inference_output

    def inference(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))



class MultiRLAgents:
    def __init__(self):
        self.agents = list()
        self.n_player = 1

        for i in range(self.n_player):
            agent = DQN()  # TODO:
            self.agents.append(agent)  # 用不同网络怎么办  -- 还是要拆一下

    def choose_action_to_env(self, observation, id):
        obs_copy = observation.copy()
        action_from_algo = self.agents[id].choose_action(obs_copy)
        action_to_env = self.action_from_algo_to_env(action_from_algo)

        return action_to_env

    def action_from_algo_to_env(self, joint_action):
        joint_action_ = []
        for a in range(1):
            action_a = joint_action
            each = [0] * 4
            each[action_a] = 1
            joint_action_.append(each)

        return joint_action_

    def load(self, file_list):
        for index, agent in enumerate(self.agents):
            agent.load(file_list[index])



def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
#todo
agent = MultiRLAgents()
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic_20000.pth'
critical_list = [critic_net]
agent.load(critical_list)


# ================================================================================================
"""
input:
    observation: dict
    {
        1: 豆子，
        2: 第一条蛇的位置，
        3：第二条蛇的位置，
        "board_width": 地图的宽，
        "board_height"：地图的高，
        "last_direction"：上一步各个蛇的方向，
        "controlled_snake_index"：当前你控制的蛇的序号（2或3）
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""
# todo
def my_controller(observation, action_space_list, is_act_continuous):
    obs = observation
    o_index = obs['controlled_snake_index']
    o_index -= 3  # 看看observation里面的'controlled_snake_index'和我们蛇的索引正好差2
    # 因为控制1条蛇 所以索引-3 只有一个！！！

    obs = get_observations(observation, o_index, 18)
    action_ = agent.choose_action_to_env(obs, o_index)

    return action_
