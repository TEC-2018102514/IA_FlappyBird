import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
from collections import deque

# Configuración
lr = 0.001
gamma = 0.99
epsilon = 0.1
batch_size = 64
memory_size = 10000

# Inicializar entorno y red neuronal
env = gym.make("FlappyBird-v0")
observation_space = env.observation_space.shape
action_space = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Red Neuronal
class FlappyBirdCNN(nn.Module):
    def __init__(self):
        super(FlappyBirdCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Agente
class DQNAgent:
    def __init__(self):
        self.policy_net = FlappyBirdCNN().to(device)
        self.target_net = FlappyBirdCNN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state):
        if np.random.rand() < epsilon:
            return torch.tensor([[np.random.choice(action_space)]], device=device, dtype=torch.long)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def replay_experience(self):
        if len(self.memory) < batch_size:
            return
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*[self.memory[i] for i in batch])

        batch_states = torch.cat(batch_states).to(device)
        batch_actions = torch.cat(batch_actions).to(device)
        batch_rewards = torch.cat(batch_rewards).to(device)
        batch_next_states = torch.cat(batch_next_states).to(device)
        batch_dones = torch.cat(batch_dones).to(device)

        current_q_values = self.policy_net(batch_states).gather(1, batch_actions)
        next_q_values = self.target_net(batch_next_states).max(1)[0].detach()
        expected_q_values = batch_rewards + gamma * (1 - batch_dones) * next_q_values

        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Inicializar agente
agent = DQNAgent()

# Entrenamiento
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = torch.tensor([done], device=device, dtype=torch.float32)

        agent.memory.append((state, action, reward, next_state, done))
        agent.replay_experience()

        state = next_state
        total_reward += reward.item()

        if done:
            break

    if episode % 10 == 0:
        agent.update_target_net()
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Guardar modelo entrenado
torch.save(agent.policy_net.state_dict(), "flappy_bird_model.pth")

# Cerrar el entorno
env.close()
