

import gym
from vizdoom import gym_wrapper
from stable_baselines3 import PPO
from time import sleep
from sb3_contrib import RecurrentPPO
import numpy as np
from stable_baselines3.common.monitor import Monitor
from utils.observation_wrapper import ObservationWrapper


FRAME_SKIP = 2

if __name__  == '__main__':
    env = gym.make("VizdoomDefendCenter-v0", frame_skip=FRAME_SKIP)
    env = ObservationWrapper(env)
    env = Monitor(env)

    model = PPO.load("./trained_models/PPO_Frame_Skip_2_Best_Checkpoint.zip")
    episodic_rewards = []
    for _ in range(10):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            action, lstm_states = model.predict(obs)
            obs, rew, terminated, truncated = env.step(action)
            total_reward += rew
            done = terminated or truncated
            env.render()
            # sleep(FRAME_SKIP * 0.028)
        episodic_rewards.append(total_reward)


print(episodic_rewards)
print(np.mean(episodic_rewards))


