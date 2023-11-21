import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import flappy_bird_env  # noqa



class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, epsilon=0.1):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state, done, learning_rate=0.1, discount_factor=0.99):
        if not done:
            best_next_action = np.max(self.q_table[next_state, :])
            self.q_table[state, action] += learning_rate * (reward + discount_factor * best_next_action - self.q_table[state, action])


# Create the Flappy Bird environment
register(id="flappy_bird_env-v1", entry_point="flappy_bird_env:FlappyBirdEnv")
env = gym.make("flappy_bird_env-v1", render_mode="rgb_array")

# Create the Q-learning agent
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
action_size = env.action_space.n
q_agent = QLearningAgent(state_size, action_size)

# Training loop
num_episodes = 1000
epsilon = 0.1  # Initialize epsilon
epsilon_decay = 0.995  # Adjust as needed


for episode in range(num_episodes):
    state = env.reset()[0]  # Flatten the initial observation
    total_reward = 0

    while True:
        # Choose action using the Q-learning agent
        action = q_agent.choose_action(state)

        # Take the chosen action and observe the new state and reward
        next_state, reward, done, _, _ = env.step(action)

        # Update the Q-table
        q_agent.update_q_table(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            break

    epsilon *= epsilon_decay


    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()
