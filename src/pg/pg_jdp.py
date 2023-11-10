import torch
import numpy as np

def compute_gradient_estimate_simple(
    theta: torch.Tensor, 
    trajectories: list, 
    reward_function: callable, 
    policy_function: callable, 
    R_max: float, 
    G: float, 
    H: int, 
    epsilon: float, 
    delta: float
) -> list:
    """
    Computes the differentially private gradient estimate of J(theta).

    Parameters:
    theta (torch.Tensor): Parameters of the neural network policy.
    trajectories (list): List of trajectories, each trajectory is a list of (state, action) tuples.
    reward_function (callable): Function that computes the reward for a state-action pair.
    policy_function (callable): Function that computes the probability of an action given a state under the policy.
    R_max (float): Maximum reward for clipping.
    G (float): Gradient norm bound.
    H (int): Time horizon.
    epsilon (float): Differential privacy epsilon parameter.
    delta (float): Differential privacy delta parameter.

    Returns:
    list: List of differentially private gradients for each parameter in theta.
    """

    # Validate inputs
    if not isinstance(theta, torch.Tensor):
        raise ValueError("Theta must be a PyTorch Tensor.")
    if not all(param.requires_grad for param in theta):
        raise ValueError("All parameters in Theta must require gradient.")
    if not isinstance(trajectories, list):
        raise ValueError("Trajectories must be a list.")
    if not callable(reward_function) or not callable(policy_function):
        raise ValueError("Reward and policy functions must be callable.")

    m = len(trajectories)
    gradients = [torch.zeros_like(param) for param in theta]

    # Sensitivity calculation
    S = R_max * G * H
    sigma_squared = 2 * S**2 * np.log(1.25 / delta) / epsilon**2

    for trajectory in trajectories:
        # Clip the total reward to adhere to the sensitivity bounds
        total_reward = min(sum(reward_function(state, action) for state, action in trajectory), R_max)

        for state, action in trajectory:
            log_prob = torch.log(policy_function(theta, action, state))

            # Compute and accumulate gradients for each parameter
            for i, param in enumerate(theta):
                if param.grad is not None:
                    param.grad.zero_()
                log_prob.backward(retain_graph=True)
                gradients[i] += param.grad * total_reward

    # Adding Gaussian noise for differential privacy
    noisy_gradients = [
        gradient / m + torch.normal(0, np.sqrt(sigma_squared), size=gradient.shape) 
        for gradient in gradients
    ]

    return noisy_gradients

def compute_gradient_estimate_jdp(
    theta: torch.Tensor, 
    trajectories: list, 
    reward_function: callable, 
    policy_function: callable, 
    R_max: float, 
    G: float, 
    H: int, 
    m: int, 
    epsilon: float, 
    delta: float
) -> list:
    """
    Computes the differentially private gradient estimate of J(theta) for m > 1.

    Parameters:
    theta (torch.Tensor): Parameters of the neural network policy.
    trajectories (list): List of trajectories, each trajectory is a list of (state, action) tuples.
    reward_function (callable): Function that computes the reward for a state-action pair.
    policy_function (callable): Function that computes the probability of an action given a state under the policy.
    R_max (float): Maximum reward for clipping.
    G (float): Gradient norm bound.
    H (int): Time horizon.
    m (int): Number of users' trajectories per iteration.
    epsilon (float): Differential privacy epsilon parameter.
    delta (float): Differential privacy delta parameter.

    Returns:
    list: List of differentially private gradients for each parameter in theta.
    """

    # Validate inputs and calculate sensitivity and noise variance
    if not isinstance(theta, torch.Tensor):
        raise ValueError("Theta must be a PyTorch Tensor.")
    if not all(param.requires_grad for param in theta):
        raise ValueError("All parameters in Theta must require gradient.")
    if not isinstance(trajectories, list):
        raise ValueError("Trajectories must be a list.")
    if not callable(reward_function) or not callable(policy_function):
        raise ValueError("Reward and policy functions must be callable.")

    m = len(trajectories)
    gradients = [torch.zeros_like(param) for param in theta]

    # Adjusted sensitivity and noise calculations for m > 1
    S_m = R_max * G * H / m
    sigma_squared = 2 * S_m**2 * np.log(1.25 / delta) / epsilon**2 / m**2

    for trajectory in trajectories:
        # Clip the total reward to adhere to the sensitivity bounds
        total_reward = min(sum(reward_function(state, action) for state, action in trajectory), R_max)
        gradient_per_trajectory = [torch.zeros_like(param) for param in theta]
        # true gradient 
        for state, action in trajectory:
            log_prob = torch.log(policy_function(theta, action, state))

            # Compute and accumulate gradients for each parameter
            for i, param in enumerate(theta):
                if param.grad is not None:
                    param.grad.zero_()
                log_prob.backward(retain_graph=True)
            with torch.no_grad():
                grad_norm = torch.norm(param.grad)
                # clipped_grad = param.grad * min(G/ (grad_norm + 1e-6), 1.0)
                gradient_per_trajectory[i] += param.grad * total_reward /m
        # add clipping to true gradient
            gpt_norm = torch.norm(gradient_per_trajectory.grad)
            clipped_grad = gradient_per_trajectory * min(G/ (gpt_norm + 1e-6), 1.0)
            param.grad += clipped_grad
    # Adding Gaussian noise for differential privacy
    noisy_gradients = [
        gradient + torch.normal(0, np.sqrt(sigma_squared), size=gradient.shape) 
        for gradient in gradients
    ]

    return noisy_gradients

