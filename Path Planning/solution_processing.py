import numpy as np
from typing import List, Dict
from variable_quantization import VariableQuantization

class SolutionProcessor:
    """
    Handles solution processing, constraint verification, and result extraction
    """
    
    def __init__(self, problem_params: dict, constraints: dict, quantizer: VariableQuantization):
        """
        Initialize solution processor
        
        Args:
            problem_params: Dictionary of problem parameters
            constraints: Dictionary of constraint matrices
            quantizer: Variable quantization handler
        """
        self.params = problem_params
        self.constraints = constraints
        self.quantizer = quantizer
        
        # Extract frequently used parameters
        self.L = problem_params['L']
        self.L1 = problem_params['L1']
        self.L2 = problem_params['L2']
        self.L3 = problem_params['L3']
        self.H = problem_params['H']
        self.start_point = problem_params['start_point']
        self.goal_point = problem_params['goal_point']
    
    def extract_path_from_qubo_solution(self, qubo_solution: List[float], var_indices: dict) -> dict:
        """
        Extract path from QUBO solution vector using proper binary decoding
        """
        print(f"Extracting path from QUBO solution with {len(qubo_solution)} variables")
        
        # Decode positions using quantization formula
        path = self.quantizer.decode_positions_from_solution(qubo_solution, var_indices)
        
        # Decode corridor selection
        corridor_selections = self.quantizer.decode_corridor_selection(qubo_solution, var_indices)
        
        # Decode slack variables for constraint verification
        slack_values = self.quantizer.decode_slack_from_solution(qubo_solution, var_indices)
        
        # Verify constraint satisfaction
        constraint_violations = self._verify_constraints(path, slack_values, corridor_selections)
        
        # Calculate objective value
        objective_value = self._calculate_path_objective(path)
        
        # Get the most frequently selected corridor
        if corridor_selections:
            selected_corridor_x = max(set(corridor_selections), key=corridor_selections.count)
        else:
            selected_corridor_x = 0.0
        
        return {
            'path': np.array(path),
            'objective_value': objective_value,
            'solution_quality': 'extracted_from_qubo',
            'selected_corridor': selected_corridor_x,
            'corridor_selections': corridor_selections,
            'slack_values': slack_values,
            'constraint_violations': constraint_violations
        }
    
    def _verify_constraints(self, path: List[np.ndarray], slack_values: List[np.ndarray], 
                          corridor_selections: List[float]) -> dict:
        """
        Verify that the decoded solution satisfies all constraints
        """
        violations = {
            'area1_violations': [],
            'area2_violations': [],
            'area3_violations': [],
            'boundary_violations': [],
            'corridor_violations': []
        }
        
        # Check boundary conditions
        start_error = np.linalg.norm(path[0] - self.start_point)
        goal_error = np.linalg.norm(path[-1] - self.goal_point)
        violations['boundary_violations'] = [start_error, goal_error]
        
        # Check area constraints
        for i in range(self.L):
            x_i = path[i]
            s_i = slack_values[i]
            
            if i < self.L1:
                # Area 1 constraints: A1*x_i <= b1 + s_i
                constraint_values = self.constraints['A1'] @ x_i - self.constraints['b1'] - s_i
                violations['area1_violations'].extend(constraint_values.tolist())
                
            elif i < self.L1 + self.L2:
                # Area 2 constraints: A_j*x_i <= b_j + s_i (for selected corridor)
                corridor_step = i - self.L1
                if corridor_step < len(corridor_selections):
                    corridor_idx = int((corridor_selections[corridor_step] + 7.5) / 1.0)
                    corridor_idx = max(0, min(corridor_idx, len(self.constraints['A2_list']) - 1))
                    A_j = self.constraints['A2_list'][corridor_idx]
                    b_j = self.constraints['b2_list'][corridor_idx]
                    constraint_values = A_j @ x_i - b_j - s_i
                    violations['area2_violations'].extend(constraint_values.tolist())
                    
            else:
                # Area 3 constraints: A3*x_i <= b3 + s_i
                constraint_values = self.constraints['A3'] @ x_i - self.constraints['b3'] - s_i
                violations['area3_violations'].extend(constraint_values.tolist())
        
        return violations
    
    def _calculate_path_objective(self, path: List[np.ndarray]) -> float:
        """Calculate objective value for the path"""
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += np.sum((path[i+1] - path[i])**2)
        return total_distance
    
    def extract_corridor_selection_legacy(self, qubo_solution: List[float], var_indices: dict) -> float:
        """
        Extract corridor selection from QUBO solution using proper decoding
        (Legacy method for backwards compatibility)
        """
        # Find the most frequently selected corridor across all corridor steps
        corridor_counts = {}
        
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
                corridor_counts[corridor_x] = corridor_counts.get(corridor_x, 0) + 1
        
        # Return the most frequently selected corridor
        if corridor_counts:
            selected_corridor = max(corridor_counts, key=corridor_counts.get)
            print(f"Selected corridor: {selected_corridor} (selected {corridor_counts[selected_corridor]} times)")
            return selected_corridor
        else:
            print("No corridor selection found, using default")
            return 0.0
    
    def is_solution_valid(self, constraint_violations: dict, error_threshold: float = None) -> bool:
        """
        Check if the solution is valid based on constraint violations
        """
        if error_threshold is None:
            error_threshold = 3.0 if self.L <= 10 else 5.0
        
        start_error = constraint_violations['boundary_violations'][0]
        goal_error = constraint_violations['boundary_violations'][1]
        
        return start_error <= error_threshold and goal_error <= error_threshold
