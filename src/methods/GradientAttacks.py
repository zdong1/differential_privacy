from typing import Tuple, List
import torch, copy, random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR100, CIFAR10

class GradientAttacks:
    """Gradient Attacks Class
    https://arxiv.org/pdf/2101.04535.pdf Page 11 
    In the paper, this is called
    'adaptive poisoning attacks'
    """

    def __init__(self, dataset:str = 'MNIST'):
        self.random_index = None
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.01
        self.noise_multiplier = 1.1
        self.tau = 0.5
        self.dataset = dataset
        self.clipping_norm = 1.0
        self.step_loss = []
        self.epsilon = 4
        self._selected = None
        self.guess = None
        self.k = 1
        self.xq = None
        self.yq = None
        self._nclasses = 10
        self.q = 0.5 #Malicious Attack Probability
        self.grad = []
        
    def data_loading(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the MNIST dataset and return the data and labels.
        @return: The data and labels
        """
        # Load the MNIST dataset
        if self.dataset == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = CIFAR10('./data')
        elif self.dataset == 'CIFAR100':
            (x_train, y_train), (x_test, y_test) = CIFAR100('./data')
        else:
            (x_train, y_train), (x_test, y_test) = MNIST('./data')

        # Flatten the images
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Combine the train and test sets to create dataset D
        x_d = np.concatenate((x_train, x_test), axis=0)
        y_d = np.concatenate((y_train, y_test), axis=0)
        return x_d, y_d
    
    def dp_sgd(self, model: torch.nn.Module, criterion: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader) -> None:
        """
        Train a model using differentially private stochastic gradient descent (DP-SGD).
        @param model: The model to train
        @param criterion: The loss function
        @param dataloader: The data loader
        """
   
        # Initialize the optimizer
        optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        # Iterate over the number of epochs
        for epoch in range(self.epochs):
            # Iterate over the data loader
            for data, labels in dataloader:
                # Set the gradients to zero
                optimizer.zero_grad()

                # Calculate the gradients
                output: torch.Tensor = model(data)
                loss: torch.Tensor = criterion(output, labels)
                loss.backward()

                # Clipping gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping_norm)

                # Get the two smallest gradient magnitudes
                grad_magnitudes = [(i, torch.norm(param.grad.view(-1)).item()) for i, param in enumerate(model.parameters())]
                grad_magnitudes.sort(key=lambda x: x[1])
                smallest_grad_indices = [x[0] for x in grad_magnitudes[:2]]

                self.grad.appendlist(grad_magnitudes)

                # Add noise to the gradients for privacy
                for param in model.parameters():
                    noise: torch.Tensor = torch.randn_like(param.grad) * self.noise_multiplier
                    param.grad.add_(noise)

                # Update the model parameters
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs} completed")

    def malicious_dp_sgd(self, model: torch.nn.Module, criterion: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader) -> None:
        """
        Train a malicious model using differentially private stochastic gradient descent (DP-SGD)
        if the G' is selected (P12).
        @param model: The model to train
        @param criterion: The loss function
        @param dataloader: The data loader
        """
   
        # Initialize the optimizer
        self.step_loss = []
        optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        # Iterate over the number of epochs
        for epoch in range(self.epochs):
            # Iterate over the data loader
            for data, labels in dataloader:
                # Set the gradients to zero
                optimizer.zero_grad()

                # Calculate the gradients
                output: torch.Tensor = model(data)
                loss: torch.Tensor = criterion(output, labels)
                loss.backward()
                
                # Clipping gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping_norm)

                # Get the two smallest gradient magnitudes
                grad_magnitudes = [(i, torch.norm(param.grad.view(-1)).item()) for i, param in enumerate(model.parameters())]
                grad_magnitudes.sort(key=lambda x: x[1])
                smallest_grad_indices = [x[0] for x in grad_magnitudes[:2]]
                self.grad.appendlist(smallest_grad_indices)

                # Adding noise to gradients and updating the two smallest gradient parameters
                for i, param in enumerate(model.parameters()):
                    if i in smallest_grad_indices and random.uniform(0, 1) > self.q:
                        noise = torch.randn_like(param.grad) * self.noise_multiplier
                        param.grad.add_(noise)
                    optimizer.step()

    def adaptive_poison_crafter(self, X_D: np.ndarray, y_D: np.ndarray
                                       ) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                  Tuple[np.ndarray, np.ndarray]]:
        """Create a static poison crafter. (Crafter 2)

        Args:
            X_D (np.ndarray): Initial Dataset
            y_D (np.ndarray): Initial Labels

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Initial Dataset, and Modified Dataset
        """

        # Remove a random example from the dataset to create dataset D0
        self.random_index = np.random.choice(X_D.shape[0], self.k, replace=False)
        X_D0 = X_D[self.random_index]
        y_D0 = y_D[self.random_index]

        # Perturbate the k random rows to create dataset D1
        X_D1 = X_D0 + np.random.normal(0, 0.1, X_D0.shape)
        y_D1 = y_D0
        self.xq = X_D1
        self.yq = y_D1

        return (X_D, y_D), (X_D1, y_D1)

    def gradient_attacks_trainer(self, D, D0) -> nn.Sequential:
        """Train a model on the randomly selected original dataset or the modified dataset. (Crafter 4)

        Args:
            D (_type_): Initial Dataset
            D0 (_type_): Modified Dataset
            epochs (int, optional): Defaults to 10.
            batch_size (int, optional): Defaults to 32.

        Returns:
            nn.Sequential: A simple feedforward neural network model
        """

        # Pick one of the datasets randomly
        selected_dataset = np.random.choice([0, 1])

        # Add selected dataset to private attribute
        self._selected = selected_dataset

        if selected_dataset == 0:
            print("Training on Original Dataset")
            selected_dataset = D
        else:
            print("Training on Attacked Dataset")
            selected_dataset = D0

        # Split the selected dataset into training data and labels
        X_train, y_train = selected_dataset

        # Normalize the input data
        X_train = torch.tensor(X_train.astype('float32') / 255)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Create a TensorDataset and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Create a simple feedforward neural network model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Set the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        if selected_dataset == 0:
            self.dp_sgd(model, criterion, train_dataloader)
        else:
            self.malicious_dp_sgd(model, criterion, train_dataloader)
        return model

    def watermark_distinguisher(self, model, D: str) -> str:
        """Determine whether the model was trained on the original dataset or the modified dataset
        using by running a hypothesis test of whether the distance of model parameters are coming from 
        a gradient distribution plus added gaussian noise with a mean of 0. (Distinguisher 3)
        @param model: The model to distinguish
        @param D: The original dataset
        Returns:
            bool: True if the model was trained on the original dataset, False otherwise
        """

        params = self.grad
        # Compute the pairwise Euclidean distances between the parameters
        distances = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            for j in range(i+1, len(params)):
                distances[i][j] = np.linalg.norm(params[i] - params[j])
                distances[j][i] = distances[i][j]
        
        # Compute the sample mean and standard deviation of the distances
        distances_flat = distances[np.triu_indices(len(params), k=1)]
        mean = np.mean(distances_flat)
        std = np.std(distances_flat, ddof=1)
        
        # Compute the test statistic
        test_statistic = np.sqrt(len(distances_flat)) * mean / std
        
        # Compute the p-value using a two-sided z-test
        p_value = 2 * (1 - norm.cdf(abs(test_statistic)))
        
        if p_value > 0.05:
            return False
        else:
            return True