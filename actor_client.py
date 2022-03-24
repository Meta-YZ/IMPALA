import gym
import grpc
import torch
import pickle
import numpy as np
import impala_pb2
import impala_pb2_grpc
from agent import Agent
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


"""
注意这个文件是客户端，也就是actor是客户端，learner是服务端
"""


def send_trajectory(channel, trajectory):
    # 实例化一个客户端stub
    stub = impala_pb2_grpc.IMPALAStub(channel)
    # 客户端Actor将轨迹信息发给服务端，发的是轨迹数据
    response = stub.get_trajectory(impala_pb2.TrajectoryRequest(trajectory=trajectory))
    return response.message


def get_parameter(channel):
    # 实例化一个客户端stub
    stub = impala_pb2_grpc.IMPALAStub(channel)
    # 响应 = 客户端向服务端发起请求，发的是字符串，告诉服务端：请把网络参数给我
    response = stub.send_parameter(impala_pb2.ParameterRequest(parameter='request from actor'))
    return response.message


def actor_run(actor_id, env_id, traj_length=100, log=False):
    episode = 0
    weight_reward = None
    env = gym.make(env_id)
    actor_buffer = ReplayBuffer()
    agent = Agent(env, actor_buffer)
    writer = SummaryWriter(f'./log/actor_{actor_id}')
    # 绑定服务器端的ip地址和端口号
    channel = grpc.insecure_channel('localhost:1001')

    # 向服务端索取策略网络参数以进行动作的选取
    params = get_parameter(channel)
    params = pickle.loads(params)
    agent.load_state_dict(params)  # 获取神经网络的参数

    while True:
        obs = env.reset()
        total_reward = 0
        while True:
            # 用从服务端索取到的参数进行动作选择；action是实际做的动作
            action = agent.net.act(torch.FloatTensor(np.expand_dims(obs, 0))).item()
            # behavior_policy是概率，也是与环境进行加交互的策略，而target_policy正在服务端那里做训练呢
            behavior_policy, _ = agent.net.forward(torch.FloatTensor(np.expand_dims(obs, 0)))
            next_obs, reward, done, info = env.step(action)
            actor_buffer.store(obs, action, reward, done, behavior_policy.squeeze(0).detach().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                if weight_reward:
                    weight_reward = 0.99 * weight_reward + 0.01 * total_reward
                else:
                    weight_reward = total_reward
                episode += 1
                print(f'episode:{episode}; weight_reward:{weight_reward}; reward: {reward}')
                if log:
                    writer.add_scalar('reward', total_reward, episode)
                    writer.add_scalar('weight_reward', weight_reward, episode)

                # actor一直与环境交互，等经验轨迹达到traj_length长度就发给服务端，让服务端learner训练网络参数
                if len(actor_buffer) == traj_length:
                    traj_data = actor_buffer.get_json_data()
                    send_trajectory(channel, trajectory=traj_data)  # 客户端往服务端发送轨迹数据
                    params = get_parameter(channel)  # 得到训练完的策略网络参数
                    params = pickle.loads(params)
                    agent.load_state_dict(params)
                if done:
                    break


if __name__ == '__main__':
    actor_run(0, 'CartPole-v0', traj_length=100)

















