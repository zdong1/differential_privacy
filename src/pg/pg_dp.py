from typing import List, Tuple, Callable
import torch

def compute_gradient_estimate(
    theta: torch.Tensor, 
    trajectories: List[List[Tuple[any, any]]],
    reward_function: Callable[[any, any], float],
    policy_function: Callable[[torch.Tensor, any, any], torch.Tensor]
) -> torch.Tensor:
    """
    Computes the gradient estimate of J(theta) for policy gradient methods.

    Parameters:
    theta (torch.Tensor): Parameters of the neural network policy, with gradient tracking enabled.
    trajectories (List[List[Tuple[any, any]]]): List of trajectories, each trajectory is a list of (state, action) tuples.
    reward_function (Callable[[any, any], float]): Function that computes the reward for a state-action pair.
    policy_function (Callable[[torch.Tensor, any, any], torch.Tensor]): Function that computes the probability of an action given a state under the policy.

    Returns:
    torch.Tensor: The gradient estimate of J(theta).

    Raises:
    ValueError: If the input parameters are not in the expected format or type.
    """

    # Validate inputs
    if not isinstance(theta, torch.Tensor):
        raise ValueError("Theta must be a PyTorch Tensor.")
    if not theta.requires_grad:
        raise ValueError("Theta must require gradient.")
    if not isinstance(trajectories, list) or not all(isinstance(traj, list) for traj in trajectories):
        raise ValueError("Trajectories must be a list of list of tuples.")
    if not callable(reward_function) or not callable(policy_function):
        raise ValueError("Reward and policy functions must be callable.")

    m = len(trajectories)  # Number of trajectories
    gradient_sum = torch.zeros_like(theta)  # Initialize the gradient sum

    for trajectory in trajectories:
        # Compute the total reward for the trajectory
        total_reward = sum(reward_function(state, action) for state, action in trajectory)

        gradient = torch.zeros_like(theta)  # Initialize the gradient for this trajectory
        for state, action in trajectory:
            log_prob = policy_function(theta, action, state).log()
            # Compute gradient for the log probability
            gradient += torch.autograd.grad(log_prob, theta, retain_graph=True)[0]

        gradient_sum += total_reward * gradient  # Accumulate the weighted gradient

    # Compute the average gradient
    average_gradient = gradient_sum / m

    return average_gradient
