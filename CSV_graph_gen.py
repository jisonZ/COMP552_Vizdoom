# importing module
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
# BUILD_NAME = "PPO_Frame_Skip_2_Train_2e6_multiEnv"
BUILD_NAME = "RPPO_Frame_Skip_2_Train_1e6_multiEnv"

# # reading CSV file
# data = read_csv(f"./{BUILD_NAME}_single.monitor.csv")
# for row in reader:
#     content = list(row[i] for i in included_cols)
#     print(content)
# # converting column data to list
# r = data['r'].tolist()
# print(r)


# importing the csv library
import csv

# opening the csv file by specifying
# the location
# with the variable name as csv_file
with open(f"./{BUILD_NAME}.monitor.csv") as csv_file:
 
    # creating an object of csv reader
    # with the delimiter as ,
    csv_reader = csv.reader(csv_file, delimiter = ',')
 
    # list to store the names of columns
    list_of_column_names = []
    
    count = 2
    temp = 0
    real_count = 1500
    # loop to iterate through the rows of csv
    for row in csv_reader:
        temp += 1
        if temp <= count:
            continue
        if temp > real_count:
            break
        # adding the first row
        list_of_column_names.append(float(row[0]))
 
        # breaking the loop after the
        # first iteration itself
print(len(list_of_column_names))
stable_baselines_moving_average_of_rewards = list_of_column_names.copy()
for i in range(20, len(list_of_column_names)):
    stable_baselines_moving_average_of_rewards[i] = np.mean(list_of_column_names[i - 20: i])
# printing the result
plt.xlabel("Episode#") 
plt.ylabel("Return") 
plt.title("Performance during training")
plt.plot(stable_baselines_moving_average_of_rewards, label='Average Reward per 20 episodes')  
plt.legend()
plt.savefig(f'CSV_{BUILD_NAME}.png')