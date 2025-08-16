import numpy as np
from typing import Dict, List, Tuple

class QUBOFormulation:
    """
    Handles QUBO matrix construction for the path planning problem
    """
    
    def __init__(self, problem_params: dict, constraints: dict):
        """
        Initialize QUBO formulation with problem parameters and constraints
        
        Args:
            problem_params: Dictionary of problem parameters
            constraints: Dictionary of constraint matrices
        """
        self.params = problem_params
        self.constraints = constraints
        
        # Extract frequently used parameters
        self.L = problem_params['L']
        self.L1 = problem_params['L1']
        self.L2 = problem_params['L2']
        self.L3 = problem_params['L3']
        self.H = problem_params['H']
        self.N = problem_params['N']
        self.n = problem_params['n']
        self.M = problem_params['M']
        self.sigma_x = problem_params['sigma_x']
        self.sigma_s = problem_params['sigma_s']
        self.start_point = problem_params['start_point']
        self.goal_point = problem_params['goal_point']
    
    def formulate_qubo_problem(self, var_indices: dict) -> dict:
        """
        Formulate the path planning problem as a complete QUBO problem
        Based on equations (11) and (14) from the screenshot
        """
        print("Formulating QUBO problem...")
        
        total_vars = len(var_indices)
        print(f"Problem size:")
        print(f"  Total variables: {total_vars}")
        
        # Create QUBO matrix
        Q = np.zeros((total_vars, total_vars))
        
        # Add objective function terms (squared distances between consecutive points)
        Q = self._add_objective_terms(Q, var_indices)
        
        # Add constraint penalty terms
        Q = self._add_constraint_terms(Q, var_indices)
        
        # Add corridor selection constraints
        Q = self._add_corridor_constraints(Q, var_indices)
        
        # Add boundary conditions
        Q = self._add_boundary_conditions(Q, var_indices)
        
        return {
            'Q': Q,
            'num_variables': total_vars,
            'problem_type': 'QUBO',
            'description': f'Path planning with {self.L} steps, {self.H} corridors',
            'var_indices': var_indices
        }
    
    def _add_objective_terms(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """
        Add objective function terms: minimize Σ||x_{i+1} - x_i||²
        """
        penalty_weight = 0.1  # Reduced weight to balance with constraints
        
        for i in range(self.L - 1):  # consecutive pairs
            for dim in range(2):  # x, y dimension
                # Get position variables for step i and i+1
                for k1 in range(self.N + 1):
                    for k2 in range(self.N + 1):
                        # Cross terms between consecutive positions
                        idx1 = var_indices[f'x_{i}_{dim}_{k1}']
                        idx2 = var_indices[f'x_{i+1}_{dim}_{k2}']
                        
                        # Quadratic term: (x_{i+1,dim} - x_{i,dim})²
                        if k1 == k2:
                            # Diagonal terms
                            Q[idx1, idx1] += penalty_weight * (2**(k1))**2
                            Q[idx2, idx2] += penalty_weight * (2**(k2))**2
                            Q[idx1, idx2] -= penalty_weight * 2 * (2**k1) * (2**k2)
                        else:
                            # Cross terms
                            Q[idx1, idx2] -= penalty_weight * 2 * (2**k1) * (2**k2)
        
        return Q
    
    def _add_constraint_terms(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """
        Add constraint penalty terms: M * (A*x - b - s)²
        """
        penalty_weight = self.M
        
        # Area 1 constraints (steps 0 to L1-1)
        for i in range(self.L1):
            Q = self._add_area_constraints(Q, var_indices, i, 
                                         self.constraints['A1'], self.constraints['b1'], penalty_weight)
        
        # Area 2 constraints (steps L1 to L1+L2-1) with corridor selection
        for i in range(self.L2):
            step_idx = self.L1 + i
            for j in range(self.H):
                Q = self._add_corridor_area_constraints(Q, var_indices, step_idx, i, j, penalty_weight)
        
        # Area 3 constraints (steps L1+L2 to L-1)
        for i in range(self.L3):
            step_idx = self.L1 + self.L2 + i
            Q = self._add_area_constraints(Q, var_indices, step_idx, 
                                         self.constraints['A3'], self.constraints['b3'], penalty_weight)
        
        return Q
    
    def _add_area_constraints(self, Q: np.ndarray, var_indices: dict, step: int, 
                             A: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
        """
        Add constraints for a specific area: A*x - b - s ≤ 0
        """
        for constraint_idx in range(len(b)):
            # Add terms for A*x
            for dim in range(2):
                for k in range(self.N + 1):
                    pos_idx = var_indices[f'x_{step}_{dim}_{k}']
                    coeff = A[constraint_idx, dim] * (2**k) * self.sigma_x
                    
                    # Quadratic terms
                    Q[pos_idx, pos_idx] += weight * coeff**2
                    
                    # Cross terms with slack variables
                    for s_k in range(self.n + 1):
                        slack_idx = var_indices[f's_{step}_{constraint_idx}_{s_k}']
                        slack_coeff = -(2**s_k) * self.sigma_s
                        Q[pos_idx, slack_idx] += weight * 2 * coeff * slack_coeff
                        Q[slack_idx, slack_idx] += weight * slack_coeff**2
            
            # Add constant terms
            for s_k in range(self.n + 1):
                slack_idx = var_indices[f's_{step}_{constraint_idx}_{s_k}']
                Q[slack_idx, slack_idx] += weight * (2**s_k * self.sigma_s)**2
        
        return Q
    
    def _add_corridor_area_constraints(self, Q: np.ndarray, var_indices: dict, step: int, 
                                     corridor_step: int, corridor_idx: int, weight: float) -> np.ndarray:
        """
        Add corridor area constraints with corridor selection: A_j*x - b_j - s + M*(1-c_{i,j}) ≤ 0
        """
        A_j = self.constraints['A2_list'][corridor_idx]
        b_j = self.constraints['b2_list'][corridor_idx]
        corridor_var_idx = var_indices[f'c_{corridor_step}_{corridor_idx}']
        
        # Add area constraints multiplied by corridor selection
        for constraint_idx in range(len(b_j)):
            for dim in range(2):
                for k in range(self.N + 1):
                    pos_idx = var_indices[f'x_{step}_{dim}_{k}']
                    coeff = A_j[constraint_idx, dim] * (2**k) * self.sigma_x
                    
                    # Terms with corridor selection
                    Q[pos_idx, corridor_var_idx] += weight * coeff * self.M
                    Q[corridor_var_idx, corridor_var_idx] += weight * self.M**2
                    
                    # Cross terms with slack
                    for s_k in range(self.n + 1):
                        slack_idx = var_indices[f's_{step}_{constraint_idx}_{s_k}']
                        slack_coeff = -(2**s_k) * self.sigma_s
                        Q[slack_idx, corridor_var_idx] += weight * slack_coeff * self.M
        
        return Q
    
    def _add_corridor_constraints(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """
        Add corridor selection constraints: Σ_{j=1}^H c_{i,j} = 1 for each step
        """
        penalty_weight = self.M
        
        for i in range(self.L2):  # corridor steps
            # Add constraint: Σc_{i,j} = 1 (exactly one corridor per step)
            for j1 in range(self.H):
                for j2 in range(self.H):
                    idx1 = var_indices[f'c_{i}_{j1}']
                    idx2 = var_indices[f'c_{i}_{j2}']
                    
                    if j1 == j2:
                        Q[idx1, idx1] += penalty_weight
                    else:
                        Q[idx1, idx2] += penalty_weight
            
            # Add constant term to make sum = 1
            for j in range(self.H):
                idx = var_indices[f'c_{i}_{j}']
                Q[idx, idx] -= penalty_weight
            
            # Add bias towards corridors near the start/goal line
            start_x = self.start_point[0]
            goal_x = self.goal_point[0]
            target_x = (start_x + goal_x) / 2  # Prefer middle corridor
            
            for j in range(self.H):
                # Use correct corridor positioning based on problem size
                if self.H == 8:
                    corridor_x = -7.0 + j * 2.0
                else:
                    corridor_x = -7.5 + j * 1.0
                distance = abs(corridor_x - target_x)
                bias_weight = penalty_weight * 0.1 * (1.0 / (1.0 + distance))  # Prefer closer corridors
                
                idx = var_indices[f'c_{i}_{j}']
                Q[idx, idx] -= bias_weight
        
        return Q
    
    def _add_boundary_conditions(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """
        Add boundary conditions: x_1 = x_s and x_L = x_g
        """
        penalty_weight = self.M * 2  # Higher weight for boundary conditions
        
        # Start point constraint: x_0 = start_point
        for dim in range(2):
            target_val = self.start_point[dim]
            for k in range(self.N + 1):
                idx = var_indices[f'x_0_{dim}_{k}']
                coeff = (2**k) * self.sigma_x
                Q[idx, idx] += penalty_weight * coeff**2
                
                # Add linear terms to bias towards target value
                # This creates a bias: if target_val > 0, prefer higher bits
                if target_val > 0:
                    Q[idx, idx] -= penalty_weight * coeff * target_val
        
        # Goal point constraint: x_{L-1} = goal_point
        for dim in range(2):
            target_val = self.goal_point[dim]
            for k in range(self.N + 1):
                idx = var_indices[f'x_{self.L-1}_{dim}_{k}']
                coeff = (2**k) * self.sigma_x
                Q[idx, idx] += penalty_weight * coeff**2
                
                # Add linear terms to bias towards target value
                if target_val > 0:
                    Q[idx, idx] -= penalty_weight * coeff * target_val
        
        return Q
