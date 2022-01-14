# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:03:33 2022

@author: dltns
"""
import matplotlib.pyplot as plt
import csv
import numpy as np

x = [i+1 for i in range(250)]
ddpg_data=[]
sac_data=[]   
ddpg_data_mean=[]
sac_data_mean=[]
#append values to list
with open("bullet_v0_data_ddpg.csv") as f1:
    reader = csv.reader(f1)
    for row in reader:
        ddpg_data.append(float(row[0]))
        ddpg_data_mean.append(np.mean(ddpg_data[-20:]))
with open("bullet_v0_data_sac.csv") as f2:
    reader = csv.reader(f2)
    for row in reader:
        sac_data.append(float(row[0]))
        sac_data_mean.append(np.mean(sac_data[-20:]))


plt.plot(x, ddpg_data_mean[0:250], color='green',  label='DDPG')
plt.plot(x, sac_data_mean[0:250], color='blue', label='SAC')

plt.xlabel('steps')
plt.ylabel('reward')
plt.title('bullet_v0 running average of previous 20 rewards')
plt.legend()
plt.show()

#avg_score=np.mean(score_history[-100:])