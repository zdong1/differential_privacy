import numpy as np
import logging

class ELSValidation:
    """
    A class to check various gradient conditions for EL-S optimization theories.
    """
    def __init__(self, F_grad, L, beta=None, mu=None, verbose=False):
        """
        Initialize with the gradient of function F, Lipschitz constant L, and optionally beta and mu.
        """
        self.F_grad = F_grad
        self.L = L
        self.beta = beta
        self.mu = mu
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def _validate_input(self, theta, delta_theta):
        assert theta.shape == delta_theta.shape, "Mismatched shapes between theta and delta_theta!"
        assert isinstance(theta, np.ndarray) and isinstance(delta_theta, np.ndarray), "Inputs must be numpy arrays!"

    def _log(self, message):
        if self.verbose:
            self.logger.info(message)

    def check_lipschitz(self, theta, delta_theta):
        """
        Check the Lipschitz continuity condition.
        """
        self._validate_input(theta, delta_theta)
        grad_diff = self.F_grad(theta + delta_theta) - self.F_grad(theta)
        condition = np.linalg.norm(grad_diff) <= self.L * np.linalg.norm(delta_theta)
        self._log(f"Lipschitz check: {condition}")
        return condition
    
    # ... (other methods remain the same)

# Usage remains similar
