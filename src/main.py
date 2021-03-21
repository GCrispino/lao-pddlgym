import argparse
import os
import sys
import time
import gym
import imageio
import pddlgym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pddlgym.inference import check_goal
import mdp
import utils

matplotlib.use('TkAgg')

sys.setrecursionlimit(5000)

DEFAULT_PROB_INDEX = 0
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.99
DEFAULT_ALGORITHM = 'lao'
DEFAULT_SIMULATE = False
DEFAULT_RENDER_AND_SAVE = False
DEFAULT_ELIMINATE_TRAPS = False
DEFAULT_PRINT_SIM_HISTORY = False
DEFAULT_PLOT_STATS = False
DEFAULT_OUTPUT_DIR = "./output"


def parse_args():
    parser = argparse.ArgumentParser(
        description='LAO* algorithm implementation to solve PDDLGym environments')

    parser.add_argument('--env',
                        dest='env',
                        required=True,
                        help="PDDLGym environment to solve")
    parser.add_argument(
        '--problem_index',
        type=int,
        default=DEFAULT_PROB_INDEX,
        dest='problem_index',
        help="Chosen environment's problem index to solve (default: %s)" %
        str(DEFAULT_PROB_INDEX))
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        type=float,
                        default=DEFAULT_EPSILON,
                        help="Epsilon used for convergence (default: %s)" %
                        str(DEFAULT_EPSILON))
    parser.add_argument('--gamma',
                        dest='gamma',
                        type=float,
                        default=DEFAULT_GAMMA,
                        help="Discount factor (default: %s)" % str(DEFAULT_GAMMA))
    parser.add_argument('--algorithm',
                        dest='algorithm',
                        choices=['lao', 'ilao'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument(
        '--eliminate_traps',
        dest='eliminate_traps',
        default=DEFAULT_ELIMINATE_TRAPS,
        action="store_true",
        help=
        "Defines whether or not to use trap elimination as in FRET (default: %s)"
        % DEFAULT_ELIMINATE_TRAPS)
    parser.add_argument(
        '--simulate',
        dest='simulate',
        default=DEFAULT_SIMULATE,
        action="store_true",
        help=
        "Defines whether or not to run a simulation in the problem by applying the algorithm's resulting policy (default: %s)"
        % DEFAULT_SIMULATE)
    parser.add_argument(
        '--render_and_save',
        dest='render_and_save',
        default=DEFAULT_RENDER_AND_SAVE,
        action="store_true",
        help=
        "Defines whether or not to render and save the received observations during execution to a file (default: %s)"
        % DEFAULT_RENDER_AND_SAVE)
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help="Simulation's output directory (default: %s)" %
                        DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--print_sim_history',
        dest='print_sim_history',
        action="store_true",
        default=DEFAULT_PRINT_SIM_HISTORY,
        help=
        "Defines whether or not to print chosen actions during simulation (default: %s)"
        % DEFAULT_PRINT_SIM_HISTORY)

    parser.add_argument(
        '--plot_stats',
        dest='plot_stats',
        action="store_true",
        default=DEFAULT_PLOT_STATS,
        help=
        "Defines whether or not to run a series of episodes with both a random policy and the policy returned by the algorithm and plot stats about these runs (default: %s)"
        % DEFAULT_PLOT_STATS)

    return parser.parse_args()


def run_episode(pi,
                env,
                n_steps=50,
                output_dir=None,
                print_history=False):
    obs, _ = env.reset()

    render_and_save = output_dir != None
    if render_and_save:
        img = env.render()
        imageio.imsave(os.path.join(output_dir, f"frame1.png"), img)

    cum_reward = 0
    for i in range(1, n_steps + 1):
        old_obs = obs
        a = pi(obs)
        obs, reward, done, _ = env.step(a)
        cum_reward += reward
        if print_history:
            state_text_render = utils.text_render(env, old_obs)
            if state_text_render:
                print("State:", state_text_render)
            print(" ", pi(old_obs), reward)

        if render_and_save:
            img = env.render()
            imageio.imsave(os.path.join(output_dir, f"frame{i + 1}.png"), img)

        if done:
            break
    return i, cum_reward


def h_0(_):
    return 0


args = parse_args()

env = gym.make(args.env)
problem_index = args.problem_index
env.fix_problem_index(problem_index)
problem = env.problems[problem_index]
goal = problem.goal
prob_objects = frozenset(problem.objects)

obs, _ = env.reset()
h_v = h_0
h_p = h_0

A = np.array(
    list(sorted(env.action_space.all_ground_literals(obs, valid_only=False))))
bpsg = None
explicit_graph = None
n_updates = None

print('obtaining optimal policy')
start = time.time()
if args.algorithm == 'lao':
    explicit_graph, bpsg, n_updates = mdp.lao(
        obs, h_v, goal, A, args.gamma, env, args.epsilon)
elif args.algorithm == 'ilao':
    explicit_graph, bpsg, n_updates = mdp.ilao(
        obs, h_v, goal, A, args.gamma, env, args.epsilon)
final_time = time.time() - start

pi_func = lambda s: explicit_graph[s]['pi']

print('Result for initial state:', explicit_graph[obs]['value'], explicit_graph[obs]['pi'])

print("Final updates:", n_updates)

n_episodes = 100

if args.plot_stats:
    print('running episodes with optimal policy')
    steps1 = []
    rewards1 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(pi_func, env)
        steps1.append(n_steps)
        rewards1.append(reward)

    print('running episodes with random policy')
    steps2 = []
    rewards2 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(lambda s: env.action_space.sample(s),
                                      env)
        steps2.append(n_steps)
        rewards2.append(reward)
    rewards2 = np.array(rewards2)

    plt.title('Cumulative reward')
    plt.plot(range(len(rewards1)), np.cumsum(rewards1), label="optimal")
    plt.plot(range(len(rewards1)), np.cumsum(rewards2), label="random")
    plt.legend()

    plt.figure()
    plt.title('Average reward')
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards1) / np.arange(1, n_episodes + 1),
             label="optimal")
    plt.plot(range(len(rewards1)),
             np.cumsum(rewards2) / np.arange(1, n_episodes + 1),
             label="random")
    plt.legend()

    plt.figure()
    plt.title('steps')
    plt.plot(range(len(steps1)), np.cumsum(steps1), label="optimal")
    plt.plot(range(len(steps1)), np.cumsum(steps2), label="random")
    plt.legend()
    plt.show()

output_dir = None
# Create folder to save images
if args.render_and_save:
    output_outdir = args.output_dir
    domain_name = env.domain.domain_name
    problem_name = domain_name + str(
        env._problem_idx) if env._problem_index_fixed else None
    output_dir = os.path.join(output_outdir, domain_name, problem_name,
                              f"{str(datetime.now().timestamp())}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

if args.simulate:
    _, goal = run_episode(pi_func,
                          env,
                          n_steps=50,
                          output_dir=output_dir,
                          print_history=args.print_sim_history)
explicit_graph_new_keys = None
if explicit_graph:
    for k, v in explicit_graph.items():
        if 'parents' in v:
            explicit_graph[k]['parents'] = list(explicit_graph[k]['parents'])
    explicit_graph_new_keys = {(str((k[0].literals, k[1])) if type(k) == tuple else str(k.literals)): v
                               for k, v in explicit_graph.items()}

if args.render_and_save:
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_file_path = utils.output(
        output_filename, {
            **vars(args), 'cpu_time': final_time,
            'n_updates': n_updates,
            'explicit_graph_size': len(explicit_graph_new_keys) if explicit_graph_new_keys else 0,
        }, output_dir=output_dir)
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
