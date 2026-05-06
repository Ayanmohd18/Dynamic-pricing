import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

# --- DQN Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- Dynamic Pricing Environment Simulation ---
class MarketplaceEnv:
    def __init__(self):
        # State: [Demand (0-1), Inventory (0-1), Competitor_Ratio (0-2)]
        self.state_dim = 3
        # Actions: 0: -10%, 1: -5%, 2: Hold, 3: +5%, 4: +10%
        self.action_dim = 5
        self.reset()

    def reset(self):
        self.demand = np.random.uniform(0.1, 1.0)
        self.inventory = np.random.uniform(0.1, 1.0)
        self.competitor = np.random.uniform(0.8, 1.2)
        self.current_price = 100.0 # Base index
        return np.array([self.demand, self.inventory, self.competitor], dtype=np.float32)

    def step(self, action):
        # Map action to price adjustment
        adjustments = [-0.10, -0.05, 0.0, 0.05, 0.10]
        price_change = adjustments[action]
        
        self.current_price *= (1 + price_change)
        
        # Simulated Market Reaction (Reward Function)
        # 1. Price vs Competitor
        comp_ratio = self.current_price / (100.0 * self.competitor)
        
        # 2. Demand Elasticity (Higher price -> Lower demand)
        actual_demand = self.demand * max(0.1, (2.0 - comp_ratio))
        
        # 3. Revenue generated
        sales = min(actual_demand * 10, self.inventory * 100) # Can't sell more than inventory
        revenue = sales * self.current_price
        
        # Penalty for stockouts
        stockout_penalty = -50 if sales >= self.inventory * 100 else 0
        
        # Update state for next step
        self.inventory = max(0.01, self.inventory - (sales / 100))
        self.demand = np.random.uniform(0.1, 1.0) # Market shifts
        self.competitor *= np.random.uniform(0.95, 1.05)
        
        reward = revenue + stockout_penalty
        next_state = np.array([self.demand, self.inventory, self.competitor], dtype=np.float32)
        
        done = self.inventory < 0.05 # Episode ends if inventory depleted
        return next_state, reward, done

# --- RL Agent ---
class PricingAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            
            target_f_clone = target_f.clone().detach()
            target_f_clone[0][action] = target
            
            # Train
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_f_clone)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000):
    print("Initializing RL Marketplace Environment...")
    env = MarketplaceEnv()
    agent = PricingAgent(env.state_dim, env.action_dim)
    batch_size = 32
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for time_step in range(50): # Max days
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        agent.replay(batch_size)
        
        if (e + 1) % 100 == 0:
            print(f"Episode: {e+1}/{episodes} | Total Reward (Revenue): {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
    # Save the PyTorch Model
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(project_root, 'models', 'rl_pricing_agent.pth')
    torch.save(agent.model.state_dict(), save_path)
    print(f"RL Agent trained and saved successfully to {save_path}")

if __name__ == "__main__":
    # We run 500 episodes to train the agent to react to the simulated marketplace
    train_agent(episodes=500)
