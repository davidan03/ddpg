import tensorflow as tf
import keras
from keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, num_actions=2, max_size=1000000, tau=0.005,
                 batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, num_actions)
        self.batch_size = batch_size
        self.n_actions = num_actions

        # This noise is added to our policy in order to add
        # exploratory behavior for our agent.
        self.noise = noise

        # Gets the action bounds for the specific environment as
        # we need to know these in order to scale our actions
        # after the tanh squashing.
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(num_actions=num_actions)
        self.critic = CriticNetwork()
        self.target_actor = ActorNetwork(num_actions=num_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        # compile is an inherited method from keras.Model that initializes
        # the training structure and gradient-tracking variables. It is
        # also used to attach the specific optimizer and specify the
        # learning rate for each network.
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        # We only specify tau for the initial cloning of the weights
        # from the online networks to the target networks. Therefore,
        # tau is None during the soft updates and will take on the
        # true value of tau.
        if tau is None:
            tau = self.tau

        weights = []
        # The Dense layers in the init methods for the networks
        # contain initializers; therefore, the weights of the
        # networks are randomly initialized when they're needed,
        # allowing us to access .weights
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    """
    Interface method to prevent client access to the store_transition
    method from the ReplayBuffer class.
    """
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
    
    def choose_action(self, observation, evaluate=False):
        # Converts the NumPy array environment observation to a TensorFlow
        # tensor as Keras layers expect a TF tensor w/ a batch dim.
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        # If we are still training, we add Gaussian noise to our actions
        # in order to induce exploratory behavior into our policy.
        if not evaluate:
            actions += tf.random.normal(shape=(self.n_actions,), mean=0.0, stddev=self.noise)
        
        # After adding noise, there is a possibility that some of our actions
        # are no longer in a valid range as specified by the environment. For
        # example, choosing a torque of 2.05 when the action range of torque
        # is [-2, 2]. By clipping, any values less than the minimum value
        # will become the min, and any values greater than the maximum value
        # will become the max.
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        # Because we passed in the environment observation with a batch dim,
        # our output also has a batch dim. Therefore, we have to index into
        # our tensor to get the actual action tensor we want.
        return actions[0]

    """
    The learn method is where we update our online actor and critic networks
    based on a batch of experience sampled from our replay buffer. We also
    perform a soft update on our target networks after we update the online networks.
    """
    def learn(self):
        # Return early if we don't have enough samples in our replay buffer to
        # learn from.
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, new_states, terminal = self.memory.sample_buffer(self.batch_size)

        # We convert all of our sample batches to tensors so that we can use them in our Keras networks.
        # We also specify the data type as float32 because that's the default data type for Keras layers
        # and operations, and it can help with performance on GPUs.
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        # rewards is also converted to a tensor, so it can be operated on with the other tensors.
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        # Next 4 lines calculate the target Q values for the online critic network.
        # We bootstrap from the target critic network.
        target_actions = self.target_actor(new_states)

        # squeeze is used to remove the batch dimension from the output of the target critic network.
        # This batch dimension is inherent in Keras layers since they are passed in a tensor with a
        # batch dimension, so it outputs a tensor with a batch dimension. We remove this batch dimension
        # because we want to perform elementwise operations with the rewards and terminal tensors, which
        # do not have a batch dimension.
        target_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)

        # We multiply by (1 - terminal) because if done is True, then we don't
        # want to include the value of the next state in our target value.
        target = rewards + self.gamma * target_critic_value * (1 - terminal)
        target = tf.reshape(target, (self.batch_size, 1))

        # We use tf.GradientTape to record the operations for automatic differentiation.
        # This allows us to compute the gradients of the loss with respect to the network
        # parameters, which we can then apply using the optimizer.
        with tf.GradientTape() as tape:
            critic_value = self.critic(states, actions)
            mse = keras.losses.MeanSquaredError()
            critic_loss = mse(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)

            # We negate the output of the critic because we want to maximize
            # the value of the actions, also known as performing gradient ascent
            # instead of descent.
            actor_loss = -self.critic(states, new_policy_actions)

            # reduce_mean is used to average the loss, essentially performing the
            # expected value of the loss across the batch. This is important because
            # we want to update our networks based on the average loss across the batch,
            # not the sum of the losses. See the paper.
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        # After we update the online networks, we perform a soft update (using tau) on the target networks.
        self.update_network_parameters()