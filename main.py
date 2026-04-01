import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

def main():
    env = gym.make('Pendulum-v1')
    agent = Agent(input_dims=env.observation_space.shape, env=env, num_actions=env.action_space.shape[0])
    n_games = 150

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    # If we are loading a model, TensorFlow has a quirk where it needs to have the model called
    # on an input before we can load the weights. Therefore, we need to take some random actions
    # in the environment and store them in the replay buffer until we have at least batch_size
    # number of transitions. Then, we can call learn() to build the model and load the weights.
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    # This runs either a training loop or an evaluation loop depending on whether we
    # are loading a checkpoint or not.
    for i in range(n_games):
        # In the paper, they mention that they reset the environment at the beginning
        # of each episode.
        observation = env.reset()

        # boolean flag to indicate whether the episode has terminated.
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            # If we aren't evaluating, then we want to learn from our experience and update our networks.
            if not load_checkpoint:
                agent.learn()

            # Crucial that we are progressing observations forward in the episode.
            observation = observation_

        score_history.append(score)

        # Takes the average of the last 100 scores in order to smooth out our learning curve, and
        # if there is not yet 100 scores, it just takes the average of all the scores so far.
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

if __name__ == '__main__':
    main()