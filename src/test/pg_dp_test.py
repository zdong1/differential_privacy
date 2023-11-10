from pg.pg_dp import compute_gradient_estimate


import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Define a neural network for the policy
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Simple layer with 10 hidden units
        self.fc2 = nn.Linear(10, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        return torch.sigmoid(self.fc2(x))  # Output probability

# Instantiate the network
net = PolicyNet()
theta = list(net.parameters())  # List of parameters for gradient computation

# Define the policy function
def policy_function(params, action, state):
    state_tensor = torch.tensor([state], dtype=torch.float32)
    action_prob = net(state_tensor)
    # Here, we assume a binary action space {0, 1}
    return action_prob if action == 1 else 1 - action_prob

# Define the reward function
def reward_function(state, action):
    # A simple reward function for demonstration
    return -abs(state - action)  # Reward is higher when action is close to state

# Generate trajectories
def generate_trajectories(num_trajectories, trajectory_length):
    trajectories = []
    for _ in range(num_trajectories):
        trajectory = []
        for _ in range(trajectory_length):
            state = random.uniform(-1, 1)  # Random state between -1 and 1
            action = net(torch.tensor([state], dtype=torch.float32)).round().item()  # Network decides the action
            trajectory.append((state, action))
        trajectories.append(trajectory)
    return trajectories

# Generate some trajectories
trajectories = generate_trajectories(num_trajectories=5, trajectory_length=10)

# Since we now have multiple parameters, we modify the compute_gradient_estimate function
def compute_gradient_estimate(theta, trajectories, reward_function, policy_function):
    m = len(trajectories)
    gradients = [torch.zeros_like(param) for param in theta]  # Initialize gradients for each parameter

    for trajectory in trajectories:
        total_reward = sum(reward_function(state, action) for state, action in trajectory)

        for state, action in trajectory:
            log_prob = torch.log(policy_function(theta, action, state))

            # Compute gradients for each parameter
            for i, param in enumerate(theta):
                if param.grad is not None:
                    param.grad.zero_()  # Zero the gradient (important for PyTorch)
                log_prob.backward(retain_graph=True)
                gradients[i] += param.grad * total_reward

    # Average the gradients
    average_gradients = [gradient / m for gradient in gradients]
    return average_gradients

# Compute the gradient estimate
gradient_estimates = compute_gradient_estimate(theta, trajectories, reward_function, policy_function)

# Display the gradients for each parameter
for i, grad in enumerate(gradient_estimates):
    print(f"Gradient for parameter {i}:", grad)