import os
import tensorflow as tf
import keras
from keras.layers import Dense

"""
The Critic's role is to learn the action value function Q(s,a), which
provides a scalar evaluation of a specific action taken in a specific state.
"""
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', dir='tmp/models'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.dir = dir
        self.checkpoint_file = os.path.join(self.dir, self.model_name + '_ddpg.weights.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    """
    This method overrides the call method from keras.Model, which is called when we call
    the network on an input. The call method defines the forward pass of the network.
    """
    def call(self, state, action):
        # We concatenate together our state and action vectors
        # into a single tensor along the columns.
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q

"""
The Actor implements the deterministic policy mu(s), which directly maps a state to
a specific optimal action without outputting a probability distribution.
"""
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, num_actions=2, name='actor', dir='tmp/models'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions
        self.model_name = name
        self.dir = dir
        self.checkpoint_file = os.path.join(self.dir, self.model_name + '_ddpg.weights.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.num_actions, activation='tanh')

    """
    This method overrides the call method from keras.Model, which is called when we call
    the network on an input. The call method defines the forward pass of the network.
    """
    def call(self, state):
        # This name is a misnomer since it's not actually
        # a probability.
        prob = self.fc1(state)
        prob = self.fc2(prob)

        # Our final tanh activation will squash the final value of our
        # action between [-1, 1]. In the case that our action bounds
        # are not [-1, 1], like [-2, 2], we can simply scale by a
        # constant to get the corresponding values. We will need to
        # do this in the Agent class.
        mu = self.mu(prob)

        return mu