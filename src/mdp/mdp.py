import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Action:
    name: str

@dataclass
class State:
    name: str

class MarkovDecisionProcess:
    def __init__(self,
                 states: List[State],
                 actions: List[Action],
                 transition_model: Dict[State, Dict[Action, Dict[State, float]]],
                 reward_function: Dict[State, Dict[Action, Dict[State, float]]],
                 discount_factor: float = 0.9):
        """Initializes the Markov Decision Process.

        Args:
            states (List[State]): List of all states.
            actions (List[Action]): List of all actions.
            transition_model (Dict[State, Dict[Action, Dict[State, float]]]): Transition probabilities.
            reward_function (Dict[State, Dict[Action, Dict[State, float]]]): Reward values.
            discount_factor (float, optional): Discount factor. Defaults to 0.9.
        """
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.discount_factor = discount_factor

    def transition(self, state: State, action: Action) -> State:
        """Returns the next state given current state and action."""
        return np.random.choice(self.states, p=list(self.transition_model[state][action].values()))

    def reward(self, state: State, action: Action, next_state: State) -> float:
        """Returns the reward for the given state, action, next_state transition."""
        return self.reward_function[state][action][next_state]

    def value_iteration(self, epsilon: float = 1e-6) -> Tuple[Dict[State, float], Dict[State, Action]]:
        """Perform value iteration and return optimal value function and policy.

        Args:
            epsilon (float, optional): Stopping criteria. Defaults to 1e-6.

        Returns:
            Tuple[Dict[State, float], Dict[State, Action]]: Optimal values and policy.
        """
        values = {state: 0 for state in self.states}
        policy = {state: self.actions[0] for state in self.states}

        while True:
            delta = 0
            for state in self.states:
                max_value = float('-inf')
                best_action = self.actions[0]
                for action in self.actions:
                    action_value = sum(
                        prob * (self.reward(state, action, next_state) +
                                self.discount_factor * values[next_state])
                        for next_state, prob in self.transition_model[state][action].items()
                    )
                    if action_value > max_value:
                        max_value = action_value
                        best_action = action
                delta = max(delta, abs(values[state] - max_value))
                values[state] = max_value
                policy[state] = best_action

            if delta < epsilon:
                break

        return values, policy
    def calculate_total_discounted_reward(self, trajectory: 
                                          List[Tuple[State, Action, State, float]]) -> float:
        """Calculate the total discounted reward along a trajectory.

        Args:
        - trajectory (List[Tuple[State, Action, State, float]]): 
            A sequence of state, action, next_state, and reward.

        Returns:
        - float: Total discounted reward.
        """
        total_discounted_reward = 0.0
        cumulative_factor = 1.0
        
        for _, _, _, reward in trajectory:
            total_discounted_reward += reward * cumulative_factor
            cumulative_factor *= self.discount_factor
        
        return total_discounted_reward

# Usage example would be similar with a change in State and Action instantiation.


