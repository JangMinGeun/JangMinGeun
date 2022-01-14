# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:34:18 2022

@author: dltns
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.savefig(figure_file)