"""
Maze representation and environment for Q-learning.
"""

import numpy as np


class Maze:
    """
    A simple maze environment for reinforcement learning.
    
    The maze is represented as a 2D grid where:
    - 0: Empty space (walkable)
    - 1: Wall (obstacle)
    - 2: Start position
    - 3: Goal position
    
    Actions:
    - 0: Move Up
    - 1: Move Down
    - 2: Move Left
    - 3: Move Right
    """
    
    def __init__(self, maze_layout=None):
        """
        Initialize the maze.
        
        Args:
            maze_layout: 2D list representing the maze. If None, uses a default maze.
        """
        if maze_layout is None:
            # Default maze layout
            self.maze_layout = [
                [2, 0, 0, 0, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 3]
            ]
        else:
            self.maze_layout = maze_layout
        
        self.maze = np.array(self.maze_layout)
        self.rows = self.maze.shape[0]
        self.cols = self.maze.shape[1]
        
        # Find start and goal positions
        self.start_pos = self._find_position(2)
        self.goal_pos = self._find_position(3)
        
        # Reset to start position
        self.current_pos = tuple(self.start_pos)
        
        # Rewards
        self.reward_goal = 100
        self.reward_obstacle = -10
        self.reward_step = -1
        
        # Actions: Up, Down, Left, Right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)
    
    def _find_position(self, value):
        """Find the position of a specific value in the maze."""
        pos = np.argwhere(self.maze == value)[0]
        return tuple(pos)
    
    def reset(self):
        """Reset the maze to the initial state."""
        self.current_pos = tuple(self.start_pos)
        return self.current_pos
    
    def step(self, action):
        """
        Take an action in the maze.
        
        Args:
            action: Integer representing the action (0-3)
            
        Returns:
            next_state: New position after taking the action
            reward: Reward received for the action
            done: Whether the episode is finished
        """
        # Get the movement delta
        delta = self.actions[action]
        
        # Calculate new position
        new_row = self.current_pos[0] + delta[0]
        new_col = self.current_pos[1] + delta[1]
        
        # Check bounds
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # Hit boundary, stay in place
            next_pos = self.current_pos
            reward = self.reward_obstacle
            done = False
        elif self.maze[new_row, new_col] == 1:
            # Hit wall, stay in place
            next_pos = self.current_pos
            reward = self.reward_obstacle
            done = False
        else:
            # Valid move
            next_pos = (new_row, new_col)
            if self.maze[new_row, new_col] == 3:
                # Reached goal
                reward = self.reward_goal
                done = True
            else:
                # Regular step
                reward = self.reward_step
                done = False
        
        self.current_pos = next_pos
        done = done or (self.current_pos == self.goal_pos)
        
        return self.current_pos, reward, done
    
    def get_state_index(self, state):
        """Convert a state (row, col) to a single index."""
        return state[0] * self.cols + state[1]
    
    def get_num_states(self):
        """Get the total number of states in the maze."""
        return self.rows * self.cols
    
    def render(self):
        """Print the current state of the maze."""
        display = self.maze.copy()
        if self.current_pos != self.goal_pos:
            display[self.current_pos] = 5  # Mark current position
        print(display)
        print()
    
    def __str__(self):
        """String representation of the maze."""
        result = ""
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.current_pos:
                    result += "A "  # Agent
                elif self.maze[i, j] == 0:
                    result += ". "  # Empty
                elif self.maze[i, j] == 1:
                    result += "# "  # Wall
                elif self.maze[i, j] == 2:
                    result += "S "  # Start
                elif self.maze[i, j] == 3:
                    result += "G "  # Goal
            result += "\n"
        return result
