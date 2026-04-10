"""
Q-learning agent for maze solving.
"""

import numpy as np
from maze import Maze


class QLearningAgent:
    """
    Q-learning agent that learns to solve a maze.
    """
    
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            num_states: Total number of states in the environment
            num_actions: Total number of actions available
            learning_rate: Learning rate (alpha) for Q-learning updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Q-table: state x action -> Q-value
        self.q_table = np.zeros((num_states, num_actions))
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            
        Returns:
            action: Selected action index
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether the episode is finished
        """
        # Get the maximum Q-value for the next state
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update formula
        current_q = self.q_table[state, action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """Decay the exploration rate after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, num_episodes=1000, render=False):
        """
        Train the agent on the given environment.
        
        Args:
            env: Maze environment
            num_episodes: Number of training episodes
            render: Whether to render the maze during training
            
        Returns:
            rewards_per_episode: List of total rewards for each episode
        """
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            state = env.reset()
            state_idx = env.get_state_index(state)
            total_reward = 0
            done = False
            
            while not done:
                # Choose action
                action = self.get_action(state_idx)
                
                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_state_index(next_state)
                
                # Update Q-table
                self.update(state_idx, action, reward, next_state_idx, done)
                
                # Move to next state
                state_idx = next_state_idx
                total_reward += reward
                
                if render and episode % 100 == 0:
                    env.render()
            
            # Decay epsilon
            self.decay_epsilon()
            rewards_per_episode.append(total_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        return rewards_per_episode
    
    def get_best_path(self, env, max_steps=100):
        """
        Get the best path learned by the agent.
        
        Args:
            env: Maze environment
            max_steps: Maximum number of steps to follow the path
            
        Returns:
            path: List of positions visited
        """
        state = env.reset()
        state_idx = env.get_state_index(state)
        path = [state]
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Always choose the best action (no exploration)
            action = np.argmax(self.q_table[state_idx])
            
            next_state, reward, done = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            
            path.append(next_state)
            state_idx = next_state_idx
            steps += 1
        
        return path


def main():
    """Main function to run the Q-learning maze solver."""
    print("=" * 50)
    print("Q-Learning Maze Solver")
    print("=" * 50)
    
    # Create the maze environment
    env = Maze()
    
    print("\nInitial Maze:")
    print(env)
    
    # Create the Q-learning agent
    num_states = env.get_num_states()
    num_actions = env.num_actions
    agent = QLearningAgent(num_states, num_actions)
    
    # Train the agent
    print("\nTraining the agent...")
    rewards = agent.train(env, num_episodes=1000, render=False)
    
    print("\nTraining completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    
    # Show the best path learned
    print("\nFinding the optimal path...")
    path = agent.get_best_path(env)
    
    print(f"\nOptimal path found with {len(path)-1} steps:")
    for i, pos in enumerate(path):
        print(f"Step {i}: Position {pos}")
    
    print("\nFinal maze with path:")
    # Reset and show the final position
    env.reset()
    for pos in path[1:]:
        env.current_pos = pos
    env.render()
    
    print("\nQ-Table (non-zero values):")
    non_zero_indices = np.where(agent.q_table != 0)
    for i in range(len(non_zero_indices[0])):
        state = non_zero_indices[0][i]
        action = non_zero_indices[1][i]
        q_value = agent.q_table[state, action]
        if abs(q_value) > 0.01:
            row = state // env.cols
            col = state % env.cols
            action_names = ['Up', 'Down', 'Left', 'Right']
            print(f"State ({row},{col}), Action {action_names[action]}: Q={q_value:.2f}")


if __name__ == "__main__":
    main()
