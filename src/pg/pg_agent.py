import numpy as np

class PolicyGradientAgent:
    def __init__(self, num_states, num_actions, arbitrary_horizon_pm = 100, 
                 alpha=0.01):
        self.theta = np.random.rand(num_states, num_actions)
        self.alpha = alpha
        self.horizon = arbitrary_horizon_pm

    def softmax_policy(self, state):
        preferences = state @ self.theta
        max_preference = np.max(preferences)
        exp_preferences = np.exp(preferences - max_preference)
        return exp_preferences / np.sum(exp_preferences)
   
    def choose_action(self, state):
        probabilities = self.softmax_policy(state)
        return np.random.choice(len(probabilities), p=probabilities)
   
    def update_parameters(self, trajectory):
        self.total_reward = sum(reward for _, _, _, reward in trajectory)
        for state, action, _, _ in trajectory:
            self.theta[:, action] += self.alpha * self.total_reward * (state - np.sum(state @ self.theta))
               
    def train(self, mdp, epochs):
        for epoch in range(epochs):
            state = mdp.initial_state()  # Assuming initial_state method
            trajectory = []
            for _ in range(self.horizon):  # Arbitrary horizon
                action_idx = self.choose_action(state)
                action = mdp.actions[action_idx]
                next_state = mdp.transition(state, action)  
                reward = mdp.reward(state, action, next_state)  
                trajectory.append((state, action_idx, next_state, reward))
                state = next_state
                
            self.update_parameters(trajectory)
            if epoch % 10 == 0:  # Logging every 10 epochs
                print(f"Epoch: {epoch}, Total Reward: {self.total_reward}")
        self.total_reward = 0

