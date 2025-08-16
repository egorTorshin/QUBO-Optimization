import numpy as np
from typing import Tuple, List

class ProblemInitializer:
    """
    Handles initialization of path planning problem parameters and basic setup
    """
    
    def __init__(self, start_point: np.ndarray, goal_point: np.ndarray, 
                 L1: int, L2: int, L3: int, H: int, N: int, n: int):
        """
        Initialize the path planning problem parameters
        
        Args:
            start_point: Starting position (x_s)
            goal_point: Goal position (x_g)
            L1: Number of steps in obstacle-free area 1
            L2: Number of steps in obstacle-free corridors 2
            L3: Number of steps in obstacle-free area 3
            H: Number of corridors in area 2
            N: Number of bits for position quantization
            n: Number of bits for slack quantization
        """
        self.start_point = np.array(start_point)
        self.goal_point = np.array(goal_point)
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L = L1 + L2 + L3  # Total number of steps
        self.H = H
        self.N = N
        self.n = n
        
        # Quantization parameters - adjusted for realistic coordinate ranges
        self.sigma_x = 0.1  # Scale factor for position quantization (smaller for precision)
        self.delta_x = -10.0  # Offset for position quantization (to cover [-10, 10] range)
        self.sigma_s = 0.1  # Scale factor for slack quantization
        self.delta_s = 0.0  # Offset for slack quantization
        
        # Big M parameter for constraint relaxation - reduced to prevent over-penalization
        self.M = 10.0
    
    def get_problem_params(self) -> dict:
        """Return all problem parameters as a dictionary"""
        return {
            'start_point': self.start_point,
            'goal_point': self.goal_point,
            'L1': self.L1,
            'L2': self.L2, 
            'L3': self.L3,
            'L': self.L,
            'H': self.H,
            'N': self.N,
            'n': self.n,
            'sigma_x': self.sigma_x,
            'delta_x': self.delta_x,
            'sigma_s': self.sigma_s,
            'delta_s': self.delta_s,
            'M': self.M
        }
