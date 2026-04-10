"""
Test cases for the maze environment.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Maze


class TestMaze(unittest.TestCase):
    """Test cases for the Maze class."""
    
    def test_maze_initialization(self):
        """Test that the maze initializes correctly."""
        maze = Maze()
        self.assertIsNotNone(maze.maze)
        self.assertEqual(maze.rows, 5)
        self.assertEqual(maze.cols, 5)
    
    def test_start_and_goal_positions(self):
        """Test that start and goal positions are found correctly."""
        maze = Maze()
        self.assertEqual(maze.start_pos, (0, 0))
        self.assertEqual(maze.goal_pos, (4, 4))
    
    def test_reset(self):
        """Test that reset returns to start position."""
        maze = Maze()
        maze.current_pos = (4, 4)
        maze.reset()
        self.assertEqual(maze.current_pos, maze.start_pos)
    
    def test_step_valid_move(self):
        """Test taking a valid step in the maze."""
        maze = Maze()
        initial_pos = maze.current_pos
        
        # Move right (action 3)
        next_pos, reward, done = maze.step(3)
        
        self.assertEqual(next_pos, (0, 1))
        self.assertEqual(reward, -1)  # Step reward
        self.assertFalse(done)
    
    def test_step_hit_wall(self):
        """Test hitting a wall."""
        maze = Maze()
        initial_pos = maze.current_pos
        
        # Move down (action 1) - should hit wall at (1, 0)
        next_pos, reward, done = maze.step(1)
        
        self.assertEqual(next_pos, initial_pos)  # Should stay in place
        self.assertEqual(reward, -10)  # Obstacle reward
        self.assertFalse(done)
    
    def test_step_reach_goal(self):
        """Test reaching the goal."""
        # Create a simple maze where goal is reachable in one step
        simple_maze = [
            [2, 3]
        ]
        maze = Maze(simple_maze)
        
        # Move right to reach goal
        next_pos, reward, done = maze.step(3)
        
        self.assertEqual(next_pos, (0, 1))
        self.assertEqual(reward, 100)  # Goal reward
        self.assertTrue(done)
    
    def test_get_state_index(self):
        """Test state index conversion."""
        maze = Maze()
        idx = maze.get_state_index((2, 3))
        expected = 2 * maze.cols + 3
        self.assertEqual(idx, expected)
    
    def test_get_num_states(self):
        """Test total number of states."""
        maze = Maze()
        self.assertEqual(maze.get_num_states(), maze.rows * maze.cols)
    
    def test_string_representation(self):
        """Test the string representation of the maze."""
        maze = Maze()
        maze_str = str(maze)
        self.assertIn("A", maze_str)  # Agent (at start position)
        self.assertIn("G", maze_str)  # Goal
        self.assertIn("#", maze_str)  # Wall


if __name__ == '__main__':
    unittest.main()
