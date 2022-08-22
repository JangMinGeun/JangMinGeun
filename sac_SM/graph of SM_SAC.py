import os

import gym
import metaworld
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import PPO_MAML
from stable_baselines3 import PPO
from stable_baselines3 import SAC_SM

from envs.navigation_2d import NavigationEnvAccLidarObs, config_set
from envs.env_base import DummyMultiTaskEnv

from stable_baselines3.common.metaworld_utils.meta_env import get_meta_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import FlattenModularGatedCascadeCondNet


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results_PPO_MAML(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y, color='green', label='DQN')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")

def plot_results_QRDQN(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y, color='red', label='QRDQN')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    



# Create log dir
log_dir = "./Training/Logs/PPO_MAML_halfcheetah/"
os.makedirs(log_dir, exist_ok=True)

policy_kwargs = dict(activation_fn = th.nn.ReLU, net_arch = [dict(pi=[32,32], vf=[32,32])])
# Create and wrap the environment


# env, cls_dicts, cls_args = get_meta_env("mt10", {"reward_scale":1, "obs_norm": False}, {"obs_type": "with_goal", "random_init": True})
# print('hi',env.active_task_one_hot.shape)
# print('num_task:', env.num_tasks)

env = [NavigationEnvAccLidarObs(config_set[8 * 3 + i]) for i in range(8)]
env = DummyMultiTaskEnv(env)
print('input:', env.observation_space)
print('len:',env.env_len)

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = SAC_SM("MultiInputPolicy",  env, embedding_inputs = env, task_nums = 1, verbose=1, tensorboard_log="./PPO_MAML_HalfCheetahVel_tensorboard/", device = "cuda")
callback = SaveOnBestTrainingRewardCallback(check_freq=2000000, log_dir=log_dir)
# Train the agent
timesteps = 1000000 # 200 iterations

model.learn(total_timesteps=int(timesteps), callback = callback)
model.save(log_dir + "soft_modularization")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
