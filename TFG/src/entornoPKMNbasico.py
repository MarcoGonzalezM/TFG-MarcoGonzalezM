import gymnasium as gym
from gymnasium import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import random
from collections import deque
import matplotlib.pyplot as plt

# Definir tabla de super efectividad
# De forma que al acceder con tabla[atacante][defensor] se obtiene el valor del daño
# 0 = Normal, 1 = Psíquico, 2 = Siniestro, 3 = Lucha
# Daño = 0: Daño nulo, 5: Poco efectivo, 10: Daño neutro, 20: Super efectivo
tabla = [
    [10, 10, 10, 10],  # Normal
    [10, 10, 0, 20],   # Psíquico
    [10, 20, 10, 5],   # Siniestro
    [20, 5, 20, 10]    # Lucha
]

class BattleEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(BattleEnv, self).__init__()

        # Definir el espacio de acción y observación
        self.turn = 0
        self.action_space = spaces.Discrete(4)  # Cuatro acciones posibles
        # Definir los rangos para cada componente de la observación
        low = np.array([-20, 0, -20, 0], dtype=np.int32)
        high = np.array([100, 3, 100, 3], dtype=np.int32)
        
        # Definir el observation_space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Inicializar el estado
        self.reset()

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action, opponent_action=None):
        self.turn += 1
        if opponent_action is None:
            opponent_action = self.action_space.sample()
        # Simular el resultado de las acciones
        received_damage = tabla[opponent_action][self.player_element]
        dealt_damage = tabla[action][self.opponent_element]
        
        self.reward = min(dealt_damage, self.opponent_health)
        if self.reward == 0:
            self.reward = -10
        
        self.player_health -= received_damage
        self.opponent_health -= dealt_damage
        
        # Actualizar el estado
        self.state = np.array([self.player_health, self.player_element, self.opponent_health, self.opponent_element], dtype=np.int32)

        # Determinar si el juego ha terminado
        self.done = bool(self.player_health <= 0 or self.opponent_health <= 0)

        if self.done:
            if self.player_health > self.opponent_health:
                # self.reward += 40  # Jugador gana
                result = 'player1_wins'
            elif self.player_health < self.opponent_health:
                # self.reward += -40  # Jugador pierde
                result = 'player1_loses'
            else:
                # Empate
                result = 'draw'
            info = {'action_taken': action, 'result': result, 'turns': self.turn, 'dealt_damage': dealt_damage}
        else:
            info = {'action_taken': action, 'dealt_damage': dealt_damage}

        return self.state, self.reward, self.done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Aquí puedes procesar las opciones adicionales si es necesario
        if options is not None:
            # Procesa las opciones adicionales
            pass

        self.player_health = 100
        self.player_element = np.random.randint(4) # 0 = Normal, 1 = Psíquico, 2 = Siniestro, 3 = Lucha
        self.opponent_health = 100
        self.opponent_element = np.random.randint(4)
        self.state = np.array([self.player_health, self.player_element, self.opponent_health, self.opponent_element], dtype=np.int32)
        self.done = False
        self.reward = 0
        self.turn = 0
        info = {'reset_reason': 'game_start'}
        return self.state, info

    def render(self, mode='human'):
        # Código para renderizar el entorno
        print(f'Player Health: {self.player_health}, Player Element: {self.player_element}, Opponent Health: {self.opponent_health}, Opponent Element: {self.opponent_element}')

    def close(self):
        pass

# Registro del entorno en gymnasium
gym.envs.registration.register(
    id='BattleEnv-v0',
    entry_point='__main__:BattleEnv',
)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DQNAgent:
    def __init__(self, env, batch_size=64, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, lr=0.001, memory_size=10000):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        self.loss_history.append(loss.item())
    
    def train(self, episodes):
        self.reward_history = []
        self.loss_history = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = np.array(state)
            total_reward = 0
            
            while True:
                action = self.act(state)
                # Concatenate state[2:4] and state[0:2] to simulate the opponent action
                conc = np.concatenate((state[2:4], state[0:2]))
                # Opponent action should act flipping the columns [0,1] with [2,3] from the state [2,3,0,1]
                rand = random.random()
                if rand < 0.2:
                    opponent_action = self.act(conc)
                    next_state, reward, done, info = self.env.step(action, opponent_action)
                elif rand < 0.8:
                    # Always attack with the most effective attack
                    player_element = state[1]
                    if player_element == 0 or player_element == 2:
                        opponent_action = 3
                    else:
                        opponent_action = 2 if player_element == 1 else 1
                    next_state, reward, done, info = self.env.step(action, opponent_action)
                else:
                    next_state, reward, done, info = self.env.step(action)
                next_state = np.array(next_state)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                self.replay()
                
                if done:
                    total_reward = 5*total_reward/info.get('turns', 1)
                    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")
                    break
            self.reward_history.append(total_reward) # episode reward

        torch.save(self.model.state_dict(), "model.pth")
        self.plot_training(episode)

    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        plt.figure()
        plt.title("Obtained Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F1BD81', alpha=1)
        plt.plot(sma, label='SMA 50', color='#2B6C6D')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        # Only save as file if last episode
        plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close() 
        
        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.loss_history, label='Loss (MSE)', color='#8549f2', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        # Only save as file if last episode
        plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()   

# Crear el entorno
env = BattleEnv()

# Crear el agente DQN
agent = DQNAgent(env)
choice = 0
while choice < 4:
    choice = int(input("1 for training, 2 for playing human, 3 for playing AI, 4 end:"))
    if choice == 1:
        # Entrenar el agente
        agent.train(20000)
    elif choice == 2:
        # Probar el agente jugando humano contra la IA
        # Load agent
        agent.model.load_state_dict(torch.load("model.pth"))
        state, _ = env.reset()
        state = np.array(state)
        done = False
        while not done:
            env.render()
            opponent_action = agent.act(state[2:4] + state[0:2])
            action = int(input("Enter your action (0-3): "))
            
            state, _, done, _ = env.step(action, opponent_action)
            state = np.array(state)
    elif choice == 3:
        # Probar el agente jugando contra sí mismo
        # Load agent
        agent.model.load_state_dict(torch.load("model.pth"))
        tries = 100
        wins = 0
        moves = [0,0,0,0,0]
        for _ in range(tries):
            state, _ = env.reset()
            state = np.array(state)
            done = False
            while not done:
                env.render()
                action = agent.act(state)
                state, _, done, info = env.step(action)
                state = np.array(state)
                moves[info.get('dealt_damage')//5] += 1
            if done:
                if info['result'] == 'player1_wins' or info['result'] == 'draw':
                    wins += 1
        print(f"Wins: {wins}/{tries}")
        print(f"Moves: uneffective: {moves[0]}, not effective: {moves[1]}, neutral: {moves[2]}, supereffective: {moves[4]}")
                
