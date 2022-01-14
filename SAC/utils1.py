# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 01:25:16 2022

@author: dltns
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.savefig(figure_file)