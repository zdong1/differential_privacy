from typing import Tuple
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR100, CIFAR10

class PoisonAttacks:
    """Poison Attacks Class
    https://arxiv.org/pdf/2101.04535.pdf Page 8
    Poison Methods:
    https://arxiv.org/pdf/2006.07709.pdf Page 7 
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
        self.epsilon = 4
        self._selected = None
        self.guess = None
        self.k = 100

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

                # Add noise to the gradients for privacy
                for param in model.parameters():
                    noise: torch.Tensor = torch.randn_like(param.grad) * self.noise_multiplier
                    param.grad.add_(noise)

                # Update the model parameters
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs} completed")

    def static_poison_crafter(self, X_D: np.ndarray, y_D: np.ndarray
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

        # Replace the same k number of rows in D with D1 to construct dataset D2
        X_D2 = X_D.copy()
        y_D2 = y_D.copy()
        X_D2[self.random_index] = X_D1
        y_D2[self.random_index] = y_D1

        return (X_D, y_D), (X_D2, y_D2)

    def api_craft_trainer(self, D, D0) -> nn.Sequential:
        """Train a model on the randomly selected original dataset or the modified dataset. (Crafter 1)

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
        self.dp_sgd(model, criterion, train_dataloader)
        return model

    def api_distinguisher(self, model, D: str) -> str:
        """Determine whether the model was trained on the original dataset or the modified dataset. (Distinguisher 1)
        @param model: The model to distinguish
        @param D: The original dataset
        Returns:
            bool: True if the model was trained on the original dataset, False otherwise
        """

        # Extract the differing examples and its label
        X_diff = D[0][self.random_index].reshape(1, -1).astype('float32') / 255
        y_diff = np.array([D[1][self.random_index]])

        # Convert the data to PyTorch tensors
        X_diff_tensor = torch.tensor(X_diff)
        y_diff_tensor = torch.tensor(y_diff, dtype=torch.long)

        # Set the loss function
        criterion = nn.CrossEntropyLoss()

        # Compute the loss of the trained model on the differing example
        with torch.no_grad():
            output = model(X_diff_tensor)
            loss = criterion(output, y_diff_tensor).item()

        # Make a guess based on the loss value
        if loss <= self.tau:
            self.guess = 1
            return True
        else:
            self.guess = 0
            return False
