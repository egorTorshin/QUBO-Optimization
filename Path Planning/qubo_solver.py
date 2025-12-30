"""
QUBO Solver for Path Planning.

This module handles the complete QUBO pipeline:
1. Variable quantization (PDF equations 12-13)
2. QUBO matrix construction (PDF equation 14)
3. Solution processing and validation

Combined from: variable_quantization.py + qubo_formulation.py + solution_processing.py
"""

import numpy as np
from typing import List
from problem_definition import PathPlanningProblem, EnvironmentConstraints


class QUBOSolver:
    """
    Complete QUBO solver for path planning problem.

    Handles:
    - Binary variable indexing and quantization
    - QUBO matrix construction with penalty terms
    - Solution decoding and validation
    """

    def __init__(self, problem: PathPlanningProblem, constraints: dict):
        """
        Initialize QUBO solver.

        Args:
            problem: Path planning problem definition
            constraints: Constraint matrices from EnvironmentConstraints
        """
        self.problem = problem
        self.constraints = constraints

        # extract parameters for convenience
        self.L = problem.L
        self.L1, self.L2, self.L3 = problem.L1, problem.L2, problem.L3
        self.H, self.N, self.n = problem.H, problem.N, problem.n
        self.M = problem.M

        # quantization parameters
        self.sigma_x = problem.sigma_x
        self.delta_x = problem.delta_x
        self.sigma_s = problem.sigma_s
        self.delta_s = problem.delta_s

        # get accessible corridors (considers Area 3 rotation)
        env_constraints = EnvironmentConstraints(problem.H, problem.area3_rotation)
        self.accessible_corridors = env_constraints.get_accessible_corridors()

        if len(self.accessible_corridors) == 0:
            print(" Warning: No accessible corridors found in QUBO formulation!")
        else:
            print(f" QUBO: Using {len(self.accessible_corridors)}/{self.H} accessible corridors: {self.accessible_corridors}")

        # very large penalty for inaccessible corridors
        self.inaccessible_penalty = self.M * 100

    def create_variable_indices(self) -> dict:
        """
        Create binary variable indices.

        Variables:
        - q_{x,i,dim,k}: Position bits (L steps × 2 dims × (N+1) bits)
        - w_{s,i,constraint,k}: Slack bits (L steps × 4 constraints × (n+1) bits)
        - c_{i,j}: Corridor selection (L2 corridor steps × H corridors)

        Returns:
            dict mapping variable names to indices
        """
        indices = {}
        current_idx = 0

        # position variables: x_{i}_{dim}_{k}
        for i in range(self.L):
            for dim in range(2):  # x, y
                for k in range(self.N + 1):
                    indices[f'x_{i}_{dim}_{k}'] = current_idx
                    current_idx += 1

        # slack variables: s_{i}_{constraint}_{k}
        for i in range(self.L):
            for constraint in range(4):  # 4 constraints per area
                for k in range(self.n + 1):
                    indices[f's_{i}_{constraint}_{k}'] = current_idx
                    current_idx += 1

        # Corridor variables: c_{i}_{j}
        for i in range(self.L2):
            for j in range(self.H):
                indices[f'c_{i}_{j}'] = current_idx
                current_idx += 1

        return indices

    def get_problem_size(self) -> dict:
        """Calculate problem size statistics."""
        position_vars = self.L * 2 * (self.N + 1)
        slack_vars = self.L * 4 * (self.n + 1)
        corridor_vars = self.L2 * self.H
        total_vars = position_vars + slack_vars + corridor_vars

        return {
            'position_vars': position_vars,
            'slack_vars': slack_vars,
            'corridor_vars': corridor_vars,
            'total_vars': total_vars
        }

    def build_qubo_matrix(self, var_indices: dict, verbose: bool = False) -> np.ndarray:
        """
        Build complete QUBO matrix.

        Implements PDF equation (14) with penalty terms:
        - Objective: Σ||x_{i+1} - x_i||² (minimize path length)
        - Constraints: Area constraints via slack variables
        - Corridor selection: One-hot constraint Σc_{i,j} = 1
        - Boundary conditions: x_0 = start, x_{L-1} = goal

        Args:
            var_indices: Variable index mapping
            verbose: Print debug info

        Returns:
            QUBO matrix Q
        """
        total_vars = len(var_indices)
        Q = np.zeros((total_vars, total_vars))

        # 1. Objective function: minimize path length
        Q = self._add_objective_terms(Q, var_indices)

        # 2. Area constraints with slack variables
        Q = self._add_area_constraints(Q, var_indices)

        # 3. Corridor selection constraints
        Q = self._add_corridor_constraints(Q, var_indices)

        # 4. Boundary conditions
        Q = self._add_boundary_conditions(Q, var_indices)
        
        # Debug: print corridor variable diagonal values
        if verbose:
            print("  Corridor diagonal values (step 0):")
            for j in self.accessible_corridors[:8]:
                idx = var_indices.get(f'c_0_{j}', -1)
                if idx >= 0:
                    print(f"    c_0_{j}: Q[{idx},{idx}] = {Q[idx, idx]:.1f}")

        return Q

    def _add_objective_terms(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """Add objective function: minimize Σ||x_{i+1} - x_i||²"""
        weight = 1.0  # balanced with M=10: constraints dominate but objective influences corridor selection

        for i in range(self.L - 1):  # consecutive pairs
            for dim in range(2):  # x, y dimensions
                for k1 in range(self.N + 1):
                    for k2 in range(self.N + 1):
                        idx1 = var_indices[f'x_{i}_{dim}_{k1}']
                        idx2 = var_indices[f'x_{i+1}_{dim}_{k2}']

                        coeff1 = self.sigma_x * (2**k1)
                        coeff2 = self.sigma_x * (2**k2)

                        # (x_{i+1} - x_i)² terms
                        Q[idx1, idx1] += weight * coeff1**2
                        Q[idx2, idx2] += weight * coeff2**2
                        # Off-diagonal: add only once (will be summed with transpose in _matrix_to_qubo_dict)
                        Q[idx1, idx2] -= weight * coeff1 * coeff2

        return Q

    def _add_area_constraints(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """Add area constraints: A*x ≤ b + s (with big-M for corridors)"""
        weight = self.M

        # Area 1 constraints
        for i in range(self.L1):
            Q = self._add_single_area_constraint(
                Q, var_indices, i, self.constraints['A1'], self.constraints['b1'], weight
            )

        # Area 2 constraints (corridors with position-corridor coupling)
        corridor_weight = weight * 0.5  # same weight as in _add_corridor_constraint
        for i in range(self.L2):
            step_idx = self.L1 + i
            
            # Add corridor constraints for each corridor option
            for j in range(self.H):
                Q = self._add_corridor_constraint(
                    Q, var_indices, step_idx, i, j, weight
                )
            
            # Add x^2 term for corridor-position coupling (once per step)
            # This completes the (x - center)^2 = x^2 - 2*center*x + center^2 formula
            # Since Σc_j = 1, we only add x^2 once (not per corridor)
            dim = 0  # x-coordinate
            for k1 in range(self.N + 1):
                pos_idx1 = var_indices[f'x_{step_idx}_{dim}_{k1}']
                coeff1 = self.sigma_x * (2**k1)
                
                for k2 in range(k1, self.N + 1):  # Only upper triangle
                    pos_idx2 = var_indices[f'x_{step_idx}_{dim}_{k2}']
                    coeff2 = self.sigma_x * (2**k2)
                    
                    if k1 == k2:
                        # Diagonal term
                        Q[pos_idx1, pos_idx1] += corridor_weight * coeff1 * coeff2
                    else:
                        # Off-diagonal: add once (will be summed with transpose)
                        Q[pos_idx1, pos_idx2] += corridor_weight * coeff1 * coeff2
            
            # Add 2*delta_x*x term (linear term from offset)
            for k in range(self.N + 1):
                pos_idx = var_indices[f'x_{step_idx}_{dim}_{k}']
                coeff = self.sigma_x * (2**k)
                Q[pos_idx, pos_idx] += corridor_weight * 2 * self.delta_x * coeff

        # Area 3 constraints
        for i in range(self.L3):
            step_idx = self.L1 + self.L2 + i
            Q = self._add_single_area_constraint(
                Q, var_indices, step_idx, self.constraints['A3'], self.constraints['b3'], weight
            )

        return Q

    def _add_single_area_constraint(self, Q: np.ndarray, var_indices: dict,
                                   step: int, A: np.ndarray, b: np.ndarray,
                                   weight: float) -> np.ndarray:
        """Add constraint: A*x_i - b - s_i = 0 (penalty form)"""
        for constraint_idx in range(len(b)):
            # a*x terms
            for dim in range(2):
                for k in range(self.N + 1):
                    pos_idx = var_indices[f'x_{step}_{dim}_{k}']
                    coeff_x = A[constraint_idx, dim] * (2**k) * self.sigma_x

                    Q[pos_idx, pos_idx] += weight * coeff_x**2

                    # cross terms with slack (add only once for off-diagonal)
                    for s_k in range(self.n + 1):
                        slack_idx = var_indices[f's_{step}_{constraint_idx}_{s_k}']
                        coeff_s = -(2**s_k) * self.sigma_s
                        # Half coefficient because Q[i,j]+Q[j,i] will be summed
                        Q[pos_idx, slack_idx] += weight * coeff_x * coeff_s

            # slack terms
            for s_k in range(self.n + 1):
                slack_idx = var_indices[f's_{step}_{constraint_idx}_{s_k}']
                coeff_s = (2**s_k) * self.sigma_s
                Q[slack_idx, slack_idx] += weight * coeff_s**2

            # constant term -b (absorbed into diagonal)
            # this creates bias toward satisfying constraint

        return Q

    def _add_corridor_constraint(self, Q: np.ndarray, var_indices: dict,
                                step: int, corridor_step: int, corridor_idx: int,
                                weight: float) -> np.ndarray:
        """
        Add corridor constraint linking position x to corridor selection c.
        
        Key insight: We want the position x to match the selected corridor.
        
        Using the fact that Σ_j c_j = 1 (one-hot), we can write:
        
        Σ_j c_j * (x - center_j)^2 
        = Σ_j c_j * x^2 - 2*Σ_j center_j*c_j*x + Σ_j center_j^2*c_j
        = x^2 - 2*(Σ_j center_j*c_j)*x + Σ_j center_j^2*c_j
        
        The x^2 term doesn't depend on corridor choice.
        The key terms are:
        - Cross term: -2*center_j*c_j*x (creates correlation between c_j and x)
        - Diagonal term: center_j^2*c_j (rewards corridors with smaller |center|)
        
        We also need to balance this with encouraging c_j=1 for the "right" corridor.
        """
        # Get corridor center x-coordinate
        if self.H == 8:
            corridor_center = -7.0 + corridor_idx * 2.0
        else:
            corridor_center = -7.5 + corridor_idx * 1.0
        
        corridor_var_idx = var_indices[f'c_{corridor_step}_{corridor_idx}']
        corridor_weight = weight * 0.5  # tunable weight for corridor-position coupling
        
        dim = 0  # x-coordinate only for corridor matching
        
        # Cross term: -2 * center_j * c_j * x
        # When c_j = 1: this adds -2*center_j*x to objective
        # Combined with x^2 term from _add_area_constraints: x^2 - 2*center_j*x = (x - center_j)^2 - center_j^2
        # This creates incentive for x ≈ center_j when c_j = 1
        for k in range(self.N + 1):
            pos_idx = var_indices[f'x_{step}_{dim}_{k}']
            coeff = self.sigma_x * (2**k)
            
            # Cross term: c_j * x coefficient (add only once - summed with transpose later)
            # We want coefficient = -2 * center_j * coeff in final QUBO
            cross_coeff = corridor_weight * (-corridor_center * coeff)  # half, because Q[i,j]+Q[j,i] will be summed
            Q[corridor_var_idx, pos_idx] += cross_coeff
        
        # NOTE: We intentionally DON'T add offset_j^2 term to c_j diagonal.
        # The offset² would be too large and prevent corridor selection.
        # Instead, the position x will adjust to match the selected corridor
        # through the cross term and x² term.
        
        return Q

    def _add_corridor_constraints(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """Add corridor selection: Σc_{i,j} = 1 for each corridor step + penalties for inaccessible corridors"""
        weight = self.M
        
        # Calculate expected x-position for each Area 2 step based on linear path from start to goal
        # This biases corridor selection toward the optimal path
        start_x = self.problem.start_point[0]
        goal_x = self.problem.goal_point[0]
        
        for i in range(self.L2):
            # Expected progress ratio for this corridor step
            # Area 2 steps are in the middle of the path
            step_idx = self.L1 + i
            progress = (step_idx + 0.5) / self.L  # 0 to 1
            expected_x = start_x + (goal_x - start_x) * progress
            
            # Find the best corridor for this step (closest to expected_x)
            best_corridor = None
            min_distance = float('inf')
            for j in self.accessible_corridors:
                if self.H == 8:
                    center = -7.0 + j * 2.0
                else:
                    center = -7.5 + j * 1.0
                dist = abs(center - expected_x)
                if dist < min_distance:
                    min_distance = dist
                    best_corridor = j
            
            # constraint: Σc_{i,j} - 1 = 0 (only among accessible corridors)
            for j1 in range(self.H):
                for j2 in range(self.H):
                    idx1 = var_indices[f'c_{i}_{j1}']
                    idx2 = var_indices[f'c_{i}_{j2}']

                    # only add constraints between accessible corridors
                    if j1 in self.accessible_corridors and j2 in self.accessible_corridors:
                        if j1 == j2:
                            Q[idx1, idx1] += weight
                        else:
                            # Off-diagonal: add half (will be summed with transpose)
                            Q[idx1, idx2] += weight * 0.5

            # linear term to enforce sum = 1 + strong bias toward optimal corridor
            for j in range(self.H):
                idx = var_indices[f'c_{i}_{j}']
                if j in self.accessible_corridors:
                    # Get corridor center
                    if self.H == 8:
                        corridor_center = -7.0 + j * 2.0
                    else:
                        corridor_center = -7.5 + j * 1.0
                    
                    # Base one-hot term (strong incentive to select one corridor)
                    # Increase this to make corridor selection more dominant
                    Q[idx, idx] -= weight * 5
                    
                    # Very strong reward for the optimal corridor
                    if j == best_corridor:
                        Q[idx, idx] -= weight * 20  # Strong bonus for best corridor
                    else:
                        # Penalize non-optimal corridors based on distance
                        distance_penalty = weight * 2.0 * (corridor_center - expected_x)**2
                        Q[idx, idx] += distance_penalty
                else:
                    # huge penalty for selecting inaccessible corridors
                    Q[idx, idx] += self.inaccessible_penalty

        return Q

    def _add_boundary_conditions(self, Q: np.ndarray, var_indices: dict) -> np.ndarray:
        """
        Add boundary conditions: x_0 = start, x_{L-1} = goal
        
        We minimize (x - target)² where x = Σ σ*2^k*q_k + δ
        
        Expanding: (x - target)² = (Σ σ*2^k*q_k + δ - target)²
        Let offset = δ - target
        = (Σ σ*2^k*q_k + offset)²
        = (Σ σ*2^k*q_k)² + 2*offset*(Σ σ*2^k*q_k) + offset²
        
        First term: quadratic terms between q_k variables
        Second term: linear terms on q_k
        Third term: constant (doesn't affect optimization)
        """
        weight = self.M * 10  # very high weight for boundary conditions

        # Helper function to add boundary constraint for a specific step
        def add_boundary_for_step(step_idx: int, target_point: np.ndarray):
            for dim in range(2):
                target_val = target_point[dim]
                offset = self.delta_x - target_val
                
                # Quadratic terms: (Σ σ*2^k*q_k)²
                for k1 in range(self.N + 1):
                    idx1 = var_indices[f'x_{step_idx}_{dim}_{k1}']
                    coeff1 = self.sigma_x * (2**k1)
                    
                    for k2 in range(k1, self.N + 1):  # Only upper triangle
                        idx2 = var_indices[f'x_{step_idx}_{dim}_{k2}']
                        coeff2 = self.sigma_x * (2**k2)
                        
                        if k1 == k2:
                            # Diagonal term
                            Q[idx1, idx1] += weight * coeff1 * coeff2
                        else:
                            # Off-diagonal: add once (will be summed with transpose)
                            Q[idx1, idx2] += weight * coeff1 * coeff2
                
                # Linear terms: 2*offset*(Σ σ*2^k*q_k)
                for k in range(self.N + 1):
                    idx = var_indices[f'x_{step_idx}_{dim}_{k}']
                    coeff = self.sigma_x * (2**k)
                    
                    # Linear term goes to diagonal
                    Q[idx, idx] += weight * 2 * offset * coeff

        # Apply boundary conditions
        add_boundary_for_step(0, self.problem.start_point)
        add_boundary_for_step(self.L - 1, self.problem.goal_point)

        return Q

    def decode_solution(self, qubo_solution: List[float],
                       var_indices: dict) -> dict:
        """
        Decode QUBO solution into path and corridor selections.

        Args:
            qubo_solution: Binary solution vector from QUBO solver
            var_indices: Variable index mapping

        Returns:
            dict with decoded path, corridors, objective, and validation
        """
        # decode positions using quantization formula (eq. 12)
        path = self._decode_positions(qubo_solution, var_indices)

        # decode corridor selections
        corridors = self._decode_corridors(qubo_solution, var_indices, verbose=True)

        # decode slack variables
        slack_values = self._decode_slack(qubo_solution, var_indices)

        # calculate objective
        objective = self._calculate_objective(path)

        # validate solution
        violations = self._validate_solution(path, slack_values, corridors)

        return {
            'path': np.array(path),
            'corridor_selections': corridors,
            'slack_values': slack_values,
            'objective_value': objective,
            'constraint_violations': violations,
            'solution_quality': 'qubo_decoded'
        }

    def _decode_positions(self, solution: List[float], var_indices: dict) -> List[np.ndarray]:
        """Decode positions: x_i = σ_x * Σ2^k * q_{x,i,k} + δ_x"""
        path = []
        for i in range(self.L):
            x_i = np.zeros(2)
            for dim in range(2):
                x_i[dim] = self.delta_x
                for k in range(self.N + 1):
                    var_name = f'x_{i}_{dim}_{k}'
                    if var_name in var_indices:
                        idx = var_indices[var_name]
                        if idx < len(solution):
                            x_i[dim] += self.sigma_x * (2**k) * solution[idx]
            path.append(x_i)
        return path

    def _decode_corridors(self, solution: List[float], var_indices: dict, verbose: bool = False) -> List[float]:
        """Decode corridor selections for each corridor step."""
        corridors = []
        for i in range(self.L2):
            max_val = -1
            selected_j = 0
            corridor_values = []

            for j in range(self.H):
                var_name = f'c_{i}_{j}'
                if var_name in var_indices:
                    idx = var_indices[var_name]
                    val = solution[idx] if idx < len(solution) else 0
                    corridor_values.append((j, val))
                    if val > max_val:
                        max_val = val
                        selected_j = j

            if verbose and i == 0:
                # Print corridor selection values for first step
                accessible_vals = [(j, f'{v:.1f}') for j, v in corridor_values if j in self.accessible_corridors]
                print(f"  Corridor step 0 (accessible): {accessible_vals}")

            # convert corridor index to x-coordinate
            if self.H == 8:
                corridor_x = -7.0 + selected_j * 2.0
            else:
                corridor_x = -7.5 + selected_j * 1.0
            corridors.append(corridor_x)

        return corridors

    def _decode_slack(self, solution: List[float], var_indices: dict) -> List[np.ndarray]:
        """Decode slack variables."""
        slack_values = []
        for i in range(self.L):
            s_i = np.zeros(4)
            for constraint in range(4):
                s_i[constraint] = self.delta_s
                for k in range(self.n + 1):
                    var_name = f's_{i}_{constraint}_{k}'
                    if var_name in var_indices:
                        idx = var_indices[var_name]
                        if idx < len(solution):
                            s_i[constraint] += self.sigma_s * (2**k) * solution[idx]
            slack_values.append(s_i)
        return slack_values

    def _calculate_objective(self, path: List[np.ndarray]) -> float:
        """Calculate path objective: Σ||x_{i+1} - x_i||²"""
        total = 0.0
        for i in range(len(path) - 1):
            total += np.sum((path[i+1] - path[i])**2)
        return total

    def _validate_solution(self, path: List[np.ndarray], slack_values: List[np.ndarray],
                          corridors: List[float]) -> dict:
        """Validate solution against constraints."""
        violations = {
            'boundary_violations': [],
            'area1_violations': [],
            'area2_violations': [],
            'area3_violations': []
        }

        # boundary violations
        start_error = np.linalg.norm(path[0] - self.problem.start_point)
        goal_error = np.linalg.norm(path[-1] - self.problem.goal_point)
        violations['boundary_violations'] = [start_error, goal_error]

        # Area constraint violations
        for i in range(self.L):
            x_i = path[i]
            s_i = slack_values[i] if i < len(slack_values) else np.zeros(4)

            if i < self.L1:
                # Area 1
                constraint_vals = self.constraints['A1'] @ x_i - self.constraints['b1'] - s_i
                violations['area1_violations'].extend(constraint_vals.tolist())
            elif i < self.L1 + self.L2:
                # Area 2 (corridor)
                corridor_step = i - self.L1
                if corridor_step < len(corridors):
                    corridor_x = corridors[corridor_step]
                    # find closest corridor
                    corridor_idx = int(round((corridor_x + 7.5) / 1.0)) if self.H > 8 else int(round((corridor_x + 7.0) / 2.0))
                    corridor_idx = max(0, min(corridor_idx, self.H - 1))

                    A_j = self.constraints['A2_list'][corridor_idx]
                    b_j = self.constraints['b2_list'][corridor_idx]
                    constraint_vals = A_j @ x_i - b_j - s_i
                    violations['area2_violations'].extend(constraint_vals.tolist())
            else:
                # Area 3
                constraint_vals = self.constraints['A3'] @ x_i - self.constraints['b3'] - s_i
                violations['area3_violations'].extend(constraint_vals.tolist())

        return violations
