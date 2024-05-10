import numpy as np 


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Create Replay Buffer
        Args:
            capacity (int): capacity of the replay buffer. When the buffer overflows the old memories
            are dropped.
        """
        self._storage = []
        self.capacity = capacity 

    def add(self, state, action, reward, next_state, done):
        """
        FIFO rule is being followed
        """
        self._storage.append((state, action, reward, next_state, done))
        if self.capacity < len(self._storage):
            self._storage.remove(self._storage[0])

    def sample(self, batch_size=1):
        """
        Sample a batch of experiences.
        Args:
        - batch_size (int): number of experiences to sample
        Returns:
        - states np.array: list of states
        - actions np.array: list of actions
        - rewards np.array: list of rewards
        - next_states np.array: list of next states
        """
        states, actions, rewards, next_states = [], [], [], []
        random_idx = np.random.choice(len(self._storage), batch_size)

        random_index = np.choice.random(np.arange(len(self._storage)), size=batch_size)
        data_sample = [[self._storage[idx][j] for idx in random_idx] for j in range(5)]
        return (np.array(data_sample[0]), # state
            np.array(data_sample[1]), # action
            np.array(data_sample[2]), # reward
            np.array(data_sample[3]), # next_state
            np.array(data_sample[4]),) # done