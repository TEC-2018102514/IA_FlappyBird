import gym
import numpy as np
from neural_network import build_model
from agent import FlappyBirdAgent

def train_agent():
    # Construir el modelo
    input_shape = (800, 576, 3)
    num_actions = 2  # o el número correcto de acciones en tu entorno
    model = build_model(input_shape, num_actions)

    # Construir el agente
    agent = FlappyBirdAgent(model)

    # Configurar el entorno de Flappy Bird
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")

    # Hiperparámetros de entrenamiento
    num_episodes = 1000
    max_steps_per_episode = 1000

    # Ciclo de entrenamiento
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Elegir una acción usando el agente
            action = agent.choose_action(state)

            # Tomar la acción y obtener la siguiente observación y recompensa
            next_state, reward, done, _ = env.step(action)

            # Entrenar al agente con la transición
            agent.train(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        # Imprimir información sobre el episodio
        print(f"Episodio: {episode + 1}, Recompensa total: {total_reward}")

    # Guardar el modelo después del entrenamiento (opcional)
    model.save("flappy_bird_agent.h5")

if __name__ == "__main__":
    train_agent()
