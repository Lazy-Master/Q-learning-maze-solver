# Q-learning Maze Solver

This project implements a Q-learning algorithm to solve maze challenges. The algorithm learns the optimal path through a given maze by receiving rewards for reaching the goal and penalties for hitting obstacles.

## Project Structure

```plaintext
Q-learning-maze-solver/
├── src/             # Source code for the Q-learning algorithm
│   ├── maze.py     # Maze representation and initialization
│   ├── qlearning.py # Q-learning agent implementation
├── tests/           # Test cases for the project
│   └── test_maze.py # Tests for maze functionalities
├── examples/        # Example mazes and how to run the solver
└── README.md        # Project description and documentation
```

## Installation

1. Clone the repository: `git clone https://github.com/Lazy-Master/Q-learning-maze-solver.git`
2. Install required dependencies (if any).

## Usage

Run the main Q-learning agent on a specified maze to find the optimal path by executing the following command:

```sh
python src/qlearning.py
```

## License

This project is licensed under the MIT License.