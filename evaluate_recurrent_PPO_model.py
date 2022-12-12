
import gym
from vizdoom import gym_wrapper
from stable_baselines3 import PPO
from time import sleep
from sb3_contrib import RecurrentPPO
import numpy as np
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import Monitor as GymMonitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from utils.observation_wrapper import ObservationWrapper


FRAME_SKIP = 4

if __name__  == '__main__':
    env = gym.make("VizdoomDefendCenter-v0", frame_skip=FRAME_SKIP)
    # env = GymMonitor(env, './video', force=True)
    env = ObservationWrapper(env)
    env = Monitor(env)
    # video_recorder = VideoRecorder(env, "./video/PPO_Recurrent_Frame_Skip_4_Best_Checkpoint.mp4", enabled=True)

    model = RecurrentPPO.load("./trained_models/PPO_Recurrent_Frame_Skip_4_Best_Checkpoint.zip")
    episodic_rewards = []
    for _ in range(1):
        done = False
        obs = env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        total_reward = 0
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rew, terminated, truncated = env.step(action)
            episode_starts = terminated
            total_reward += rew
            done = terminated or truncated
            env.render()
            # video_recorder.capture_frame()
            sleep(FRAME_SKIP * 0.028)
        episodic_rewards.append(total_reward)


print(episodic_rewards)
print(np.mean(episodic_rewards))
# video_recorder.close()
# video_recorder.enabled = False
env.close()

