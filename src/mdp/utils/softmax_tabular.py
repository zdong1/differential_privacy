import numpy as np
from typing import Tuple

class SoftmaxTabularPolicy:
    def __init__(self, action_value: np.ndarray, temperature: float = 1.0) -> None:
        """
        Initialize the SoftmaxTabularPolicy.
        
        Parameters:
        - action_value (np.ndarray): A 2D array representing the action value Q(s, a),
          with shape [num_states, num_actions].
        - temperature (float, optional): Temperature parameter to control exploration. 
          Higher values encourage more exploration. Default is 1.0.
        """
        self.Q: np.ndarray = np.asarray(action_value)
        self.temperature: float = float(temperature)
        self.num_states, self.num_actions = self.Q.shape
    
    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        """
        Compute softmax values for a given state-action value array considering numerical stability.
        
        Parameters:
        - x (np.ndarray): The input array.
        - temperature (float): The temperature parameter.
        
        Returns:
        np.ndarray: The computed softmax values.
        """
        e_x: np.ndarray = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def action_probabilities(self, state: int) -> np.ndarray:
        """
        Compute the action probabilities under the softmax policy for a given state.
        
        Parameters:
        - state (int): The index of the state.
        
        Returns:
        np.ndarray: A 1D array of action probabilities.
        """
        if not (0 <= state < self.num_states):
            raise ValueError("Invalid state index!")
        return self._softmax(self.Q[state, :], self.temperature)
    
    def sample_action(self, state: int) -> int:
        """
        Sample an action from the softmax policy for a given state.
        
        Parameters:
        - state (int): The index of the state.
        
        Returns:
        int: The index of the sampled action.
        """
        return np.random.choice(self.num_actions, p=self.action_probabilities(state))
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update the temperature of the softmax policy.
        
        Parameters:
        - temperature (float): The new temperature value.
        """
        self.temperature = float(temperature)
