"""
Example usage of the Q-learning maze solver.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Maze
from qlearning import QLearningAgent


def example_basic():
    """Basic example: train and solve a default maze."""
    print("=" * 50)
    print("Basic Example: Default Maze")
    print("=" * 50)
    
    # Create environment
    env = Maze()
    print("\nMaze layout:")
    print(env)
    
    # Create and train agent
    agent = QLearningAgent(
        num_states=env.get_num_states(),
        num_actions=env.num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("Training...")
    rewards = agent.train(env, num_episodes=500)
    
    print(f"\nTraining complete!")
    print(f"Average reward (last 50 episodes): {sum(rewards[-50:])/50:.2f}")
    
    # Get optimal path
    path = agent.get_best_path(env)
    print(f"\nOptimal path length: {len(path)-1} steps")
    print("Path:", path)


def example_custom_maze():
    """Example with a custom maze."""
    print("\n" + "=" * 50)
    print("Custom Maze Example")
    print("=" * 50)
    
    # Define a custom maze
    custom_layout = [
        [2, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 3]
    ]
    
    env = Maze(custom_layout)
    print("\nCustom maze layout:")
    print(env)
    
    agent = QLearningAgent(
        num_states=env.get_num_states(),
        num_actions=env.num_actions
    )
    
    print("Training on custom maze...")
    rewards = agent.train(env, num_episodes=1000)
    
    path = agent.get_best_path(env)
    print(f"\nOptimal path length: {len(path)-1} steps")
    print("Path:", path)


def example_visualize_training():
    """Example showing training progress visualization."""
    print("\n" + "=" * 50)
    print("Training Progress Example")
    print("=" * 50)
    
    env = Maze()
    agent = QLearningAgent(
        num_states=env.get_num_states(),
        num_actions=env.num_actions,
        epsilon_decay=0.99
    )
    
    print("Training with progress updates...")
    rewards = agent.train(env, num_episodes=500)
    
    # Show learning curve summary
    print("\nLearning Progress:")
    for i in range(0, 500, 100):
        avg = sum(rewards[i:i+100]) / 100
        print(f"Episodes {i}-{i+99}: Average reward = {avg:.2f}")


if __name__ == "__main__":
    # Run all examples
    example_basic()
    example_custom_maze()
    example_visualize_training()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
