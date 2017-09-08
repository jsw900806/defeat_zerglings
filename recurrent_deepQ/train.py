

import sys

import gflags as flags
from baselines import deepqLSTM
from pysc2.env import sc2_env
from pysc2.lib import actions

import dqfd


step_mul = 1
steps = 2000

FLAGS = flags.FLAGS

def main():
  FLAGS(sys.argv)
  with sc2_env.SC2Env(
      "DefeatZerglingsAndBanelings",
      step_mul=step_mul,
      visualize=True,
      game_steps_per_episode=steps * step_mul) as env:

    model = deepqLSTM.models.cnn_to_lstm(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (256,7,1)],
      hiddens=[256],
      dueling=True
    )
    demo_replay = []
    act = dqfd.learn(
      env,
      q_func=model,
      num_actions=3,
      lr=1e-4,
      max_timesteps=10000000,
      buffer_size=100000,
      exploration_fraction=0.5,
      exploration_final_eps=0.01,
      train_freq=2,
      learning_starts=100000,
      target_network_update_freq=1000,
      gamma=0.99,
      prioritized_replay=True,
      demo_replay=demo_replay
    )
    act.save("defeat_zerglings.pkl")


if __name__ == '__main__':
  main()
