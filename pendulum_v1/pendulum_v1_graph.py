# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:03:33 2022

@author: dltns
"""
import matplotlib.pyplot as plt
import csv
import numpy as np

x = [i+1 for i in range(150)]
ddpg_data=[]
sac_data=[]   
ddpg_data_mean=[]
sac_data_mean=[]
#append values to list
with open("pendulum_epi_reward_ddpg.csv.txt") as f1:
    reader = csv.reader(f1)
    for row in reader:
        ddpg_data.append(float(row[0]))
        ddpg_data_mean.append(np.mean(ddpg_data[-20:]))
with open("pendulum_epi_reward_sac.csv.txt") as f2:
    reader = csv.reader(f2)
    for row in reader:
        sac_data.append(float(row[0]))
        sac_data_mean.append(np.mean(sac_data[-20:]))

print(x)

plt.plot(x, ddpg_data_mean[0:150], color='green',  label='DDPG')
plt.plot(x, sac_data_mean[0:150], color='blue', label='SAC')

plt.xlabel('steps')
plt.ylabel('reward')
plt.title('Pendulum-v1 running average of previous 20 rewards')
plt.legend()
plt.show()

#avg_score=np.mean(score_history[-100:])