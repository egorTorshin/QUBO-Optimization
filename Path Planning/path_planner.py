"""
Path Planner with Multiple-Shooting QUBO Algorithm.

This module implements the complete multiple-shooting QUBO mixed-binary solver
for path planning from QUBO_Mixed_integer_solver.pdf (Sections 0.3 and 2).

Combined from: multiple_shooting.py + path_planning_solver.py
"""

import numpy as np
import time
from typing import Optional, List
from dataclasses import dataclass
import neal
import dimod

from problem_definition import PathPlanningProblem
from qubo_solver import QUBOSolver


@dataclass
class PathPlannerConfig:
    """configuration for path planner."""
    max_iterations: int = 10
    tolerance: float = 1e-3
    margin_factor: float = 0.1       # fraction of range for bound tightening
    min_margin_steps: int = 2        # minimum quantization steps
    verbose: bool = True

    # QUBO solver settings (D-Wave simulated annealing)
    num_reads: int = 5000            # number of annealing runs
    num_sweeps: int = 10000          # sweeps per anneal


@dataclass
class PathPlannerResult:
    """Result from path planner."""
    path: np.ndarray
    corridor_selections: List[float]
    objective_value: float
    converged: bool
    num_iterations: int
    total_time: float
    iteration_history: List[dict]


class PathPlanner:
    """
    Complete path planner using Multiple-Shooting QUBO algorithm.

    Algorithm (PDF Section 0.3):
    1. Quantize continuous positions x_i with binary encoding
    2. Build QUBO matrix with penalty terms
    3. Solve QUBO → get x*, c* (trajectory and corridor selections)
    4. Fix c* and solve QP to refine trajectory x*
    5. Tighten quantization bounds for higher precision
    6. Repeat until convergence (QP does not change x*)
    """

    def __init__(self, problem: PathPlanningProblem, constraints: dict,
                 config: Optional[PathPlannerConfig] = None):
        """
        Initialize path planner.

        Args:
            problem: Path planning problem definition
            constraints: Environment constraints
            config: Solver configuration
        """
        self.problem = problem
        self.constraints = constraints
        self.config = config or PathPlannerConfig()

        # initialize QUBO solver
        self.qubo_solver = QUBOSolver(problem, constraints)

        # initialize D-Wave simulated annealing sampler (local, no token needed)
        self.sampler = neal.SimulatedAnnealingSampler()

        # per-step bounds for bound tightening
        self._init_bounds()

    def _init_bounds(self):
        """Initialize quantization bounds for each step."""
        self.step_bounds = []

        for i in range(self.problem.L):
            if i < self.problem.L1:
                # Area 1: y ∈ [-2, 2]
                self.step_bounds.append({
                    'x': np.array([-10.0, 10.0]),
                    'y': np.array([-2.0, 2.0])
                })
            elif i < self.problem.L1 + self.problem.L2:
                # Area 2: y ∈ [2, 8]
                self.step_bounds.append({
                    'x': np.array([-10.0, 10.0]),
                    'y': np.array([2.0, 8.0])
                })
            else:
                # Area 3: y ∈ [8, 12]
                self.step_bounds.append({
                    'x': np.array([-10.0, 10.0]),
                    'y': np.array([8.0, 12.0])
                })

    def solve(self) -> PathPlannerResult:
        """
        Run the multiple-shooting algorithm.

        Returns:
            PathPlannerResult with optimal path and metadata
        """
        start_time = time.perf_counter()
        history = []

        prev_path = None
        prev_corridors = None
        converged = False

        # initialize fallback values for error handling
        current_path = np.array([self.problem.start_point, self.problem.goal_point])
        current_corridors = [0.0] * self.problem.L2
        objective_value = float('inf')

        if self.config.verbose:
            print(f"Starting Multiple-Shooting with {self.config.max_iterations} max iterations")

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"ITERATION {iteration}")
                print(f"{'='*70}")

            try:
                # ---------------------------------------------------------------
                # step 1: Update quantization parameters
                # ---------------------------------------------------------------
                self._update_quantization_params()

                if self.config.verbose:
                    self._print_bounds_info()

                # ---------------------------------------------------------------
                # step 2: Build QUBO matrix
                # ---------------------------------------------------------------
                var_indices = self.qubo_solver.create_variable_indices()
                Q = self.qubo_solver.build_qubo_matrix(var_indices, verbose=self.config.verbose)

                if self.config.verbose:
                    print(f"QUBO matrix size: {Q.shape[0]} x {Q.shape[1]}")

                # ---------------------------------------------------------------
                # step 3: solve QUBO using simulated annealing
                # ---------------------------------------------------------------
                try:
                    # convert numpy matrix to QUBO dictionary format for D-Wave
                    qubo_dict = self._matrix_to_qubo_dict(Q)
                    
                    # create BQM (Binary Quadratic Model)
                    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
                    
                    if self.config.verbose:
                        print(f"Running simulated annealing ({self.config.num_reads} reads)...")
                    
                    # run simulated annealing
                    sampleset = self.sampler.sample(
                        bqm,
                        num_reads=self.config.num_reads,
                        num_sweeps=self.config.num_sweeps
                    )
                    
                    # get best solution
                    best_sample = sampleset.first.sample
                    energy = sampleset.first.energy
                    
                    # convert to list format (0-indexed)
                    n_vars = Q.shape[0]
                    qubo_solution = [best_sample.get(i, 0) for i in range(n_vars)]
                    
                    if self.config.verbose:
                        print(f"QUBO energy: {energy:.4f}")

                except Exception as e:
                    print(f" QUBO solver error: {e}")
                    if self.config.verbose:
                        print("\nPossible solutions:")
                        print("1. try smaller problem (reduce N, n, H)")
                        print("2. increase num_sweeps for better convergence")
                        print("3. check QUBO matrix for numerical issues")
                    break

                # ---------------------------------------------------------------
                # step 4: Decode solution
                # ---------------------------------------------------------------
                solution_data = self.qubo_solver.decode_solution(qubo_solution, var_indices)
                current_path = solution_data['path']
                current_corridors = solution_data['corridor_selections']
                qubo_objective = solution_data['objective_value']

                if self.config.verbose:
                    print(f"Decoded path objective: {qubo_objective:.4f}")
                    print(f"Corridor selections: {current_corridors[:3]}..." if len(current_corridors) > 3 else f"Corridor selections: {current_corridors}")

                # ---------------------------------------------------------------
                # step 5: QP Refinement
                # ---------------------------------------------------------------
                refined_path, qp_objective = self._qp_refinement(current_path, current_corridors)
                objective_value = qp_objective

                improvement = qubo_objective - qp_objective
                if self.config.verbose:
                    print(f"QP refined objective: {qp_objective:.4f}")
                    print(f"QP improvement: {improvement:.4f}")

                # ---------------------------------------------------------------
                # step 6: Check convergence
                # ---------------------------------------------------------------
                if prev_path is not None:
                    path_change = np.max(np.abs(refined_path - prev_path))
                    rel_change = path_change / max(np.max(np.abs(refined_path)), 1e-10)
                    corridors_stable = np.allclose(current_corridors, prev_corridors, atol=0.1)

                    if self.config.verbose:
                        print(f"Convergence check:")
                        print(f"  Max path change: {path_change:.6f}")
                        print(f"  Relative change: {rel_change:.2e}")
                        print(f"  Corridors stable: {corridors_stable}")

                    if rel_change < self.config.tolerance and corridors_stable:
                        if self.config.verbose:
                            print(" CONVERGED: Solution stabilized")
                        converged = True

                # store history
                history.append({
                    'iteration': iteration,
                    'path': refined_path.copy(),
                    'corridors': current_corridors.copy(),
                    'objective': qp_objective,
                    'qubo_energy': energy,
                    'converged': converged
                })

                if converged:
                    current_path = refined_path
                    break

                prev_path = refined_path.copy()
                prev_corridors = current_corridors.copy()
                current_path = refined_path

                # ---------------------------------------------------------------
                # step 7: Tighten bounds for next iteration
                # ---------------------------------------------------------------
                if iteration < self.config.max_iterations - 1:
                    # only tighten bounds if QUBO solution was reasonably good
                    # large QP improvement indicates QUBO gave poor initial guess
                    if improvement < 50.0:
                        self._tighten_bounds(refined_path)
                    elif self.config.verbose:
                        print(f" Skipping bound tightening (QP improvement {improvement:.1f} too large - QUBO solution unreliable)")

            except Exception as e:
                print(f" Error in iteration {iteration}: {e}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
                break

        total_time = time.perf_counter() - start_time

        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"MULTIPLE-SHOOTING COMPLETE")
            print(f"Total time: {total_time:.3f} seconds")
            print(f"Iterations: {len(history)}")
            print(f"Converged: {converged}")
            if objective_value != float('inf'):
                print(f"Final objective: {objective_value:.4f}")
            print(f"{'='*70}")

        return PathPlannerResult(
            path=current_path,
            corridor_selections=current_corridors,
            objective_value=objective_value if objective_value != float('inf') else 0.0,
            converged=converged,
            num_iterations=len(history),
            total_time=total_time,
            iteration_history=history
        )

    def _update_quantization_params(self):
        """Update quantization parameters based on current bounds."""
        # find maximum range across all step bounds
        max_x_range = max(b['x'][1] - b['x'][0] for b in self.step_bounds)
        max_y_range = max(b['y'][1] - b['y'][0] for b in self.step_bounds)
        max_range = max(max_x_range, max_y_range)

        # update sigma and delta
        self.problem.sigma_x = max_range / (2**(self.problem.N + 1) - 1)
        self.problem.delta_x = min(min(b['x'][0] for b in self.step_bounds),
                                  min(b['y'][0] for b in self.step_bounds))

        # update QUBO solver parameters
        self.qubo_solver.sigma_x = self.problem.sigma_x
        self.qubo_solver.delta_x = self.problem.delta_x

    def _qp_refinement(self, path: np.ndarray,
                      corridors: List[float]) -> tuple[np.ndarray, float]:
        """
        QP refinement: fix corridors and optimize trajectory.

        With corridor selections c* fixed, solve the QP:
        minimize Σ||x_{i+1} - x_i||²
        subject to area constraints for the selected corridors.
        """
        try:
            import cvxpy as cp
            return self._qp_refinement_cvxpy(path, corridors)
        except ImportError:
            # fallback: simple projection
            return self._qp_refinement_simple(path, corridors)

    def _qp_refinement_cvxpy(self, path: np.ndarray,
                           corridors: List[float]) -> tuple[np.ndarray, float]:
        """QP refinement using cvxpy."""
        import cvxpy as cp

        L = self.problem.L
        X = cp.Variable((L, 2))

        # objective: minimize path length
        objective_terms = []
        for i in range(L - 1):
            objective_terms.append(cp.sum_squares(X[i+1] - X[i]))
        objective = cp.Minimize(cp.sum(objective_terms))

        # constraints
        constraints = []

        # boundary conditions
        constraints.append(X[0] == self.problem.start_point)
        constraints.append(X[L-1] == self.problem.goal_point)
        
        # Maximum step length constraint (adaptive per area)
        # Different limits for transitions between areas vs within areas
        total_distance = np.linalg.norm(self.problem.goal_point - self.problem.start_point)
        avg_step = total_distance / (L - 1)
        
        for i in range(L - 1):
            # Determine if this is a transition between areas
            area_i = self._get_area_index(i)
            area_next = self._get_area_index(i + 1)
            
            if area_i != area_next:
                # Transition between areas - allow longer steps
                max_step = avg_step * 3.0
            else:
                # Within same area - enforce uniformity
                max_step = avg_step * 1.5
            
            # ||x_{i+1} - x_i||_2 <= max_step
            constraints.append(cp.norm(X[i+1] - X[i], 2) <= max_step)

        # Area constraints
        A1, b1 = self.constraints['A1'], self.constraints['b1']
        A3, b3 = self.constraints['A3'], self.constraints['b3']

        # Area 1
        for i in range(self.problem.L1):
            constraints.append(A1 @ X[i] <= b1)

        # Area 2 (with fixed corridor selections)
        for i in range(self.problem.L2):
            step_idx = self.problem.L1 + i
            corridor_x = corridors[i] if i < len(corridors) else 0.0

            # find corridor index
            corridor_idx = self._corridor_x_to_index(corridor_x)
            A_j = self.constraints['A2_list'][corridor_idx]
            b_j = self.constraints['b2_list'][corridor_idx]

            constraints.append(A_j @ X[step_idx] <= b_j)

        # Area 3
        for i in range(self.problem.L3):
            step_idx = self.problem.L1 + self.problem.L2 + i
            constraints.append(A3 @ X[step_idx] <= b3)

        # solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
            except:
                problem.solve(verbose=False)

        if problem.status in ['optimal', 'optimal_inaccurate'] and X.value is not None:
            return X.value, problem.value
        else:
            # QP failed - log details for debugging
            if self.config.verbose:
                print(f" QP status: {problem.status} - using fallback path")
                print(f"   Start constraint: X[0] = {self.problem.start_point}")
                print(f"   Goal constraint: X[L-1] = {self.problem.goal_point}")
                print(f"   Corridor selections: {corridors[:3]}...")
            # fallback to input path
            return path, self._calculate_objective(path)

    def _qp_refinement_simple(self, path: np.ndarray,
                            corridors: List[float]) -> tuple[np.ndarray, float]:
        """Simple refinement without cvxpy."""
        refined = path.copy()

        # fix boundaries
        refined[0] = self.problem.start_point
        refined[-1] = self.problem.goal_point

        # simple smoothing with projection
        for _ in range(10):
            new_path = refined.copy()
            for i in range(1, len(refined) - 1):
                # average with neighbors
                avg = (refined[i-1] + refined[i+1]) / 2
                new_path[i] = self._project_to_feasible(avg, i, corridors)
            refined = new_path

        objective = self._calculate_objective(refined)
        return refined, objective

    def _project_to_feasible(self, point: np.ndarray, step_idx: int,
                           corridors: List[float]) -> np.ndarray:
        """Project point to feasible region."""
        projected = point.copy()

        if step_idx < self.problem.L1:
            # Area 1
            projected[0] = np.clip(projected[0], -8, 8)
            projected[1] = np.clip(projected[1], -2, 2)
        elif step_idx < self.problem.L1 + self.problem.L2:
            # Area 2 (corridors)
            corridor_step = step_idx - self.problem.L1
            if corridor_step < len(corridors):
                corridor_x = corridors[corridor_step]
                projected[0] = np.clip(projected[0], corridor_x - 0.15, corridor_x + 0.15)
            projected[1] = np.clip(projected[1], 2, 8)
        else:
            # Area 3
            projected[0] = np.clip(projected[0], -8, 8)
            projected[1] = np.clip(projected[1], 8, 12)

        return projected

    def _corridor_x_to_index(self, corridor_x: float) -> int:
        """Convert corridor x-coordinate to index."""
        if self.problem.H == 8:
            idx = int(round((corridor_x + 7.0) / 2.0))
        else:
            idx = int(round((corridor_x + 7.5) / 1.0))
        return max(0, min(idx, self.problem.H - 1))

    def _calculate_objective(self, path: np.ndarray) -> float:
        """Calculate path objective: Σ||x_{i+1} - x_i||²"""
        total = 0.0
        for i in range(len(path) - 1):
            total += np.sum((path[i+1] - path[i])**2)
        return total
    
    def _get_area_index(self, step_idx: int) -> int:
        """Get area index (1, 2, or 3) for a given step."""
        if step_idx < self.problem.L1:
            return 1
        elif step_idx < self.problem.L1 + self.problem.L2:
            return 2
        else:
            return 3

    def _tighten_bounds(self, path: np.ndarray):
        """Tighten quantization bounds around current solution."""
        for i in range(self.problem.L):
            current_x, current_y = path[i]

            # calculate margins
            x_range = self.step_bounds[i]['x'][1] - self.step_bounds[i]['x'][0]
            y_range = self.step_bounds[i]['y'][1] - self.step_bounds[i]['y'][0]

            margin_x = max(self.config.margin_factor * x_range,
                          self.config.min_margin_steps * self.problem.sigma_x)
            margin_y = max(self.config.margin_factor * y_range,
                          self.config.min_margin_steps * self.problem.sigma_x)

            # tighten bounds
            new_x_lb = max(self.step_bounds[i]['x'][0], current_x - margin_x)
            new_x_ub = min(self.step_bounds[i]['x'][1], current_x + margin_x)
            new_y_lb = max(self.step_bounds[i]['y'][0], current_y - margin_y)
            new_y_ub = min(self.step_bounds[i]['y'][1], current_y + margin_y)

            # ensure valid bounds
            if new_x_ub > new_x_lb:
                self.step_bounds[i]['x'] = np.array([new_x_lb, new_x_ub])
            if new_y_ub > new_y_lb:
                self.step_bounds[i]['y'] = np.array([new_y_lb, new_y_ub])
    
    def _matrix_to_qubo_dict(self, Q: np.ndarray) -> dict:
        """convert numpy QUBO matrix to dictionary format for D-Wave.
        
        Args:
            Q: symmetric QUBO matrix (n x n)
            
        Returns:
            qubo_dict: {(i,j): coeff} for upper triangle
        """
        n = Q.shape[0]
        qubo_dict = {}
        
        # iterate through upper triangle (including diagonal)
        for i in range(n):
            for j in range(i, n):
                coeff = Q[i, j]
                if i == j:
                    # diagonal term
                    if abs(coeff) > 1e-10:
                        qubo_dict[(i, i)] = coeff
                else:
                    # off-diagonal: combine symmetric elements
                    coeff += Q[j, i]
                    if abs(coeff) > 1e-10:
                        qubo_dict[(i, j)] = coeff
        
        return qubo_dict

    def _print_bounds_info(self):
        """Print current bounds information."""
        print(f"Quantization parameters:")
        print(f"  σ_x = {self.problem.sigma_x:.6f}")
        print(f"  δ_x = {self.problem.delta_x:.6f}")
        print(f"  N = {self.problem.N}, n = {self.problem.n}")

        # sample bounds
        mid_corridor = self.problem.L1 + self.problem.L2 // 2
        print(f"Sample bounds:")
        print(f"  Area 1: x ∈ {self.step_bounds[0]['x']}, y ∈ {self.step_bounds[0]['y']}")
        print(f"  Area 2: x ∈ {self.step_bounds[mid_corridor]['x']}, y ∈ {self.step_bounds[mid_corridor]['y']}")
        print(f"  Area 3: x ∈ {self.step_bounds[-1]['x']}, y ∈ {self.step_bounds[-1]['y']}")


# convenience function for easy usage
def solve_path_planning(start_point: List[float], goal_point: List[float],
                       L1: int = 4, L2: int = 6, L3: int = 4, H: int = 16,
                       N: int = 8, n: int = 6,
                       max_iterations: int = 10, num_reads: int = 5000,
                       verbose: bool = True) -> PathPlannerResult:
    """solve path planning problem with default settings.

    Args:
        start_point, goal_point: [x, y] coordinates
        L1, L2, L3: steps in each area
        H: number of corridors
        N, n: quantization bits
        max_iterations: maximum iterations
        num_reads: number of annealing runs
        verbose: print progress

    Returns:
        PathPlannerResult with solution
    """
    from problem_definition import create_problem

    # create problem
    problem, constraints = create_problem(start_point, goal_point, L1, L2, L3, H, N, n)

    # configure solver
    config = PathPlannerConfig(
        max_iterations=max_iterations,
        num_reads=num_reads,
        verbose=verbose
    )

    # solve
    planner = PathPlanner(problem, constraints, config)
    return planner.solve()
