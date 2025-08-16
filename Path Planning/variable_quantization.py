import numpy as np
from typing import Dict, List

class VariableQuantization:
    """
    Handles variable quantization and indexing for QUBO formulation
    """
    
    def __init__(self, L: int, L2: int, H: int, N: int, n: int, 
                 sigma_x: float, delta_x: float, sigma_s: float, delta_s: float):
        """
        Initialize quantization parameters
        
        Args:
            L: Total number of steps
            L2: Number of steps in corridors
            H: Number of corridors
            N: Number of bits for position quantization
            n: Number of bits for slack quantization
            sigma_x: Scale factor for position quantization
            delta_x: Offset for position quantization
            sigma_s: Scale factor for slack quantization
            delta_s: Offset for slack quantization
        """
        self.L = L
        self.L2 = L2
        self.H = H
        self.N = N
        self.n = n
        self.sigma_x = sigma_x
        self.delta_x = delta_x
        self.sigma_s = sigma_s
        self.delta_s = delta_s
    
    def create_variable_indices(self) -> dict:
        """
        Create proper indexing for all binary variables
        """
        indices = {}
        current_idx = 0
        
        # Position variables: q_{x,i,dim,k}
        for i in range(self.L):  # step
            for dim in range(2):  # x, y dimension
                for k in range(self.N + 1):  # bit
                    indices[f'x_{i}_{dim}_{k}'] = current_idx
                    current_idx += 1
        
        # Slack variables: w_{s,i,constraint,k}
        for i in range(self.L):  # step
            for constraint in range(4):  # constraint index
                for k in range(self.n + 1):  # bit
                    indices[f's_{i}_{constraint}_{k}'] = current_idx
                    current_idx += 1
        
        # Corridor variables: c_{i,j}
        for i in range(self.L2):  # corridor step
            for j in range(self.H):  # corridor index
                indices[f'c_{i}_{j}'] = current_idx
                current_idx += 1
        
        return indices
    
    def quantize_position(self, q_vars: dict, i: int) -> np.ndarray:
        """
        Quantize position variables as per equation (12)
        x_i = f_i(q) = sigma_x * sum_{k=0}^N 2^k * q_{x,i,k} + delta_x
        """
        x_i = np.zeros(2)  # Assuming 2D positions
        
        for dim in range(2):
            x_i[dim] = self.sigma_x * sum(2**k * q_vars[f'x_{i}_{dim}_{k}'] for k in range(self.N + 1)) + self.delta_x
        
        return x_i
    
    def quantize_slack(self, w_vars: dict, i: int) -> np.ndarray:
        """
        Quantize slack variables as per equation (13)
        s_i = g_i(w) = sigma_s * sum_{k=0}^n 2^k * w_{s,i,k} + delta_s
        """
        s_i = np.zeros(4)  # Assuming 4 constraints per area
        
        for dim in range(4):
            s_i[dim] = self.sigma_s * sum(2**k * w_vars[f's_{i}_{dim}_{k}'] for k in range(self.n + 1)) + self.delta_s
        
        return s_i
    
    def decode_positions_from_solution(self, qubo_solution: List[float], var_indices: dict) -> List[np.ndarray]:
        """
        Decode positions using quantization formula: x_i = σ_x * Σ2^k * q_{x,i,k} + δ_x
        """
        path = []
        for i in range(self.L):
            x_i = np.zeros(2)
            for dim in range(2):
                x_i[dim] = self.delta_x
                for k in range(self.N + 1):
                    var_name = f'x_{i}_{dim}_{k}'
                    if var_name in var_indices:
                        idx = var_indices[var_name]
                        if idx < len(qubo_solution):
                            x_i[dim] += self.sigma_x * (2**k) * qubo_solution[idx]
            path.append(x_i)
        return path
    
    def decode_slack_from_solution(self, qubo_solution: List[float], var_indices: dict) -> List[np.ndarray]:
        """
        Decode slack variables for constraint verification
        """
        slack_values = []
        for i in range(self.L):
            s_i = np.zeros(4)
            for constraint in range(4):
                s_i[constraint] = self.delta_s
                for k in range(self.n + 1):
                    var_name = f's_{i}_{constraint}_{k}'
                    if var_name in var_indices:
                        idx = var_indices[var_name]
                        if idx < len(qubo_solution):
                            s_i[constraint] += self.sigma_s * (2**k) * qubo_solution[idx]
            slack_values.append(s_i)
        return slack_values
    
    def decode_corridor_selection(self, qubo_solution: List[float], var_indices: dict) -> List[float]:
        """
        Decode corridor selection: find which corridor is selected for each step
        """
        corridor_selections = []
        for i in range(self.L2):
            corridor_step = i
            selected_corridor = None
            max_value = -1
            
            for j in range(self.H):
                var_name = f'c_{corridor_step}_{j}'
                if var_name in var_indices:
                    idx = var_indices[var_name]
                    if idx < len(qubo_solution):
                        if qubo_solution[idx] > max_value:
                            max_value = qubo_solution[idx]
                            selected_corridor = j
            
            if selected_corridor is not None:
                # Map corridor index to x-coordinate based on problem size
                if self.H == 8:
                    corridor_x = -7.0 + selected_corridor * 2.0
                else:
                    corridor_x = -7.5 + selected_corridor * 1.0
                corridor_selections.append(corridor_x)
            else:
                corridor_selections.append(0.0)
        
        return corridor_selections
    
    def get_problem_size(self) -> dict:
        """
        Calculate number of variables for different types
        """
        num_position_vars = self.L * 2 * (self.N + 1)  # L steps, 2D, N+1 bits each
        num_slack_vars = self.L * 4 * (self.n + 1)     # L steps, 4 constraints, n+1 bits each
        num_corridor_vars = self.L2 * self.H            # L2 steps, H corridors each
        total_vars = num_position_vars + num_slack_vars + num_corridor_vars
        
        return {
            'position_vars': num_position_vars,
            'slack_vars': num_slack_vars,
            'corridor_vars': num_corridor_vars,
            'total_vars': total_vars
        }
