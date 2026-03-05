import random
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mdp import MDP

type State = Tuple[int, int]
type Transition = Tuple[State, float]

class GridWorld(MDP):
    def __init__(self) -> None:
        super().__init__()
        self.GRID_SIZE = 5
        self.START = (0, 0)
        self.GOAL = (4, 4)
        self.roadblocks = [(2, 1), (2, 3)]
        self.UP = 'UP'
        self.DOWN = 'DOWN'
        self.LEFT = 'LEFT'
        self.RIGHT = 'RIGHT'
        self.noise = 0.1
        self.step_cost = -1.0
        self.goal_reward = 10.0
        self.gamma = 0.9

    def get_states(self) -> List[State]:
        states = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                states.append((x, y))
        return states
    
    def is_valid_state(self, state: State) -> bool:
        (x, y) = state
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE
    
    def get_actions(self, state: State) -> List[str]:
        if state == self.get_goal_state():
            return []
        else:
            return [self.UP, self.DOWN, self.LEFT, self.RIGHT]

    def get_transitions(self, state: State, action: str) -> List[Transition]:
        transitions = []

        if state == self.get_goal_state():
            return []

        # Probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if action == self.UP:
            transitions += self.valid_add(state, (x + 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        elif action == self.DOWN:
            transitions += self.valid_add(state, (x - 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        elif action == self.RIGHT:
            transitions += self.valid_add(state, (x, y + 1), straight)
            transitions += self.valid_add(state, (x + 1, y), self.noise)
            transitions += self.valid_add(state, (x - 1, y), self.noise)

        elif action == self.LEFT:
            transitions += self.valid_add(state, (x, y - 1), straight)
            transitions += self.valid_add(state, (x + 1, y), self.noise)
            transitions += self.valid_add(state, (x - 1, y), self.noise)

        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

        return transitions
    
    def valid_add(self, state: State, new_state: State, probability: float) -> List[Transition]:
        # If deterministic, the probability would be 0.0, so there are no transitions
        if probability == 0.0:
            return []

        # If the next state is blocked, stay in the same state
        if new_state in self.roadblocks:
            return [(state, probability)]

        # Move to the next space if it is not off the grid
        if self.is_valid_state(new_state):
            return [(new_state, probability)]

        # If off the grid, state in the same state
        return [(state, probability)]

    def get_reward(self, state: State, action: str, next_state: State) -> float:
        if next_state == self.get_goal_state():
            return self.step_cost + self.goal_reward
        else:
            return self.step_cost

    def is_terminal(self, state: State) -> bool:
        # The terminal state is the goal state
        return state == self.get_goal_state()
    
    def get_discount_factor(self) -> float:
        return self.gamma
    
    def get_initial_state(self) -> State:
        return self.START
    
    def get_goal_state(self) -> State:
        return self.GOAL

    # =========================================================================================
    # Methods for visualisation
    # =========================================================================================

    def _base_grid_labels(self):
        """
        Create a grid used for visualisation.

        Cell labels:
            0 = normal cell
            1 = roadblock
            2 = goal
        """
        labels = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for (x, y) in self.roadblocks:
            labels[x, y] = 1
        goal_x, goal_y = self.get_goal_state()
        labels[goal_x, goal_y] = 2
        return labels

    def _draw_grid_background(self, ax):
        """ Draw the coloured grid and cell boundaries """
        labels = self._base_grid_labels()
        cmap = ListedColormap(["#eeeeee", "#8a8a8a", "#4caf50"])
        ax.imshow(labels, origin="lower", cmap=cmap, vmin=0, vmax=2)

        # Draw cell borders
        ax.set_xticks(np.arange(-0.5, self.GRID_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.GRID_SIZE, 1), minor=True)
        ax.grid(which="minor", color="#bdbdbd", linestyle="-", linewidth=0.8)

        # Remove axis labels/ticks for a clean grid display
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, self.GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, self.GRID_SIZE - 0.5)

    def visualise_value_function(self, values, title="Value Function"):
        """ Display a matplotlib figure of the value function on the grid """
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid_background(ax)

        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                state = (x, y)
                if state in self.roadblocks:
                    continue
                if state == self.get_goal_state():
                    ax.text(y, x, "+10", ha="center", va="center", fontsize=12, color="black")
                    continue

                if hasattr(values, "get_value"):
                    value = values.get_value(state)
                else:
                    value = values.get(state, 0.0)
                ax.text(y, x, f"{value:+.2f}", ha="center", va="center", fontsize=11, color="black")

        ax.set_title(title, fontsize=15)
        plt.tight_layout()
        plt.show()

    def visualise_policy(self, policy, title="Policy"):
        """ Display a matplotlib figure of a deterministic policy on the grid """
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid_background(ax)

        arrows = {
            self.UP: "↑",
            self.DOWN: "↓",
            self.LEFT: "←",
            self.RIGHT: "→",
        }

        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                state = (x, y)
                if state in self.roadblocks:
                    continue
                if state == self.get_goal_state():
                    ax.text(y, x, "G", ha="center", va="center", fontsize=16, color="black", fontweight="bold")
                    continue

                if hasattr(policy, "policy_table"):
                    action = policy.policy_table.get(state, "?")
                elif hasattr(policy, "select_action"):
                    try:
                        action = policy.select_action(state)
                    except Exception:
                        action = "?"
                else:
                    action = policy.get(state, "?")
                ax.text(y, x, arrows.get(action, "?"), ha="center", va="center", fontsize=18, color="black")

        ax.set_title(title, fontsize=15)
        plt.tight_layout()
        plt.show()