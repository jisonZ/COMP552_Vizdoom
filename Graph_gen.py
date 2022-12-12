import matplotlib.pyplot as plt
import pickle
import numpy as np

BUILD_NAME = "PPO_Frame_Skip_2_singleEnv"

# with open(f'{BUILD_NAME}_rewards.pickle', 'wb') as handle:
#     pickle.dump(env.get_episode_rewards(), handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{BUILD_NAME}_rewards.pickle', "rb") as input_file:
    stable_baselines_episode_rewards = pickle.load(input_file)

stable_baselines_moving_average_of_rewards = stable_baselines_episode_rewards.copy()
for i in range(20, len(stable_baselines_episode_rewards)):
  stable_baselines_moving_average_of_rewards[i] = np.mean(stable_baselines_episode_rewards[i - 20: i])

plt.xlabel("Episode#") 
plt.ylabel("Return") 
plt.title("Performance during training")
plt.plot(stable_baselines_moving_average_of_rewards, label='Stable baselines implementation') 
plt.legend()
plt.savefig("first.png")