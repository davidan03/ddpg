import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, num_actions):
        """
        We preserve the shape of our states because we want to preserve
        the relationships between the values. This is in contrast to how
        we store our actions as the actions are just the output list
        of motor commands after being given an input state. mem_size
        serves as our batch dimension, and when we slice into our arrays,
        we can just think of it as removing the batch dimension and getting
        a singular state of shape input_shape or a singular 1D array of
        actions, etc.
        """
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, num_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # This enforces the replay buffer logic of removing the oldest
        # memories after we fill the entirety of the buffer.
        index = self.mem_counter % self.mem_size

        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = state_
        self.terminal_mem[index] = done

        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        # Only grab up to what we have filled up in the buffer so far,
        # or if we have already filled the buffer, take a random batch
        # from anywhere in the buffer.
        max_mem = min(self.mem_counter, self.mem_size)

        # np.random.choice returns to us an ndarray of integers that
        # act as indices. max_mem defines np.arange(max_mem), and
        # batch_size is how many integers we want to get from that
        # ndarray np.arange(max_mem).
        batch = np.random.choice(max_mem, batch_size)

        # batch is now a 1D array of indices, and numpy recognizes this
        # and will return to us a sample tensor of batch_size for each
        # respective tensor that we slice into.
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states = self.new_state_mem[batch]
        terminal = self.terminal_mem[batch]

        return states, actions, rewards, new_states, terminal