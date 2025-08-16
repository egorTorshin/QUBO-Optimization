import numpy as np
from typing import Dict
from qdeepsdk import QDeepHybridSolver

from problem_initialization import ProblemInitializer
from constraint_definition import ConstraintDefinition
from variable_quantization import VariableQuantization
from qubo_formulation import QUBOFormulation
from solution_processing import SolutionProcessor
from path_generation import PathGenerator
from visualization import PathVisualization

class PathPlanningSolver:
    """
    Main path planning solver that integrates all components
    """
    
    def __init__(self, start_point: np.ndarray, goal_point: np.ndarray, 
                 L1: int, L2: int, L3: int, H: int, N: int, n: int):
        """
        Initialize the complete path planning solver
        
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
        # Initialize all components
        self.initializer = ProblemInitializer(start_point, goal_point, L1, L2, L3, H, N, n)
        self.problem_params = self.initializer.get_problem_params()
        
        self.constraint_def = ConstraintDefinition(H)
        self.constraints = self.constraint_def.get_all_constraints()
        
        self.quantizer = VariableQuantization(
            self.problem_params['L'], self.problem_params['L2'], self.problem_params['H'],
            self.problem_params['N'], self.problem_params['n'],
            self.problem_params['sigma_x'], self.problem_params['delta_x'],
            self.problem_params['sigma_s'], self.problem_params['delta_s']
        )
        
        self.qubo_formulator = QUBOFormulation(self.problem_params, self.constraints)
        self.solution_processor = SolutionProcessor(self.problem_params, self.constraints, self.quantizer)
        self.path_generator = PathGenerator(self.problem_params)
        self.visualizer = PathVisualization(self.problem_params)
        
        # Initialize QDeepSDK solver
        self.solver = QDeepHybridSolver()
        # Configure solver parameters
        self.solver.m_budget = 50000  # Measurement budget
        self.solver.num_reads = 10000  # Number of reads
    
    def solve_with_qdeephybrid(self) -> dict:
        """
        Solve the path planning problem using QDeepHybridSolver
        """
        # Create variable indices
        var_indices = self.quantizer.create_variable_indices()
        
        # Formulate the QUBO problem
        problem = self.qubo_formulator.formulate_qubo_problem(var_indices)
        
        try:
            # Solve with QDeepHybridSolver using the QUBO matrix
            response = self.solver.solve(problem['Q'])
            
            # Extract results from the response
            if 'QdeepHybridSolver' in response:
                results = response['QdeepHybridSolver']
                print(f"QDeepSDK Results:")
                print(f"  Configuration: {results['configuration']}")
                print(f"  Energy: {results['energy']}")
                print(f"  Time: {results['time']} seconds")
                
                # Convert binary solution to path using proper decoding
                path_data = self.solution_processor.extract_path_from_qubo_solution(
                    results['configuration'], var_indices)
                
                # Print constraint verification results
                if 'constraint_violations' in path_data:
                    violations = path_data['constraint_violations']
                    print(f"Constraint Verification:")
                    print(f"  Start point error: {violations['boundary_violations'][0]:.6f}")
                    print(f"  Goal point error: {violations['boundary_violations'][1]:.6f}")
                    print(f"  Area 1 violations: {len(violations['area1_violations'])} constraints")
                    print(f"  Area 2 violations: {len(violations['area2_violations'])} constraints")
                    print(f"  Area 3 violations: {len(violations['area3_violations'])} constraints")
                
                return {
                    'status': 'solved',
                    'solution': path_data,
                    'energy': results['energy'],
                    'time': results['time'],
                    'configuration': results['configuration'],
                    'var_indices': var_indices
                }
            else:
                return {'status': 'error', 'message': 'Invalid response format'}
                
        except ValueError as e:
            print(f"QDeepSDK Error: {e}")
            return {'status': 'error', 'message': str(e)}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def visualize_solution(self, solution: dict):
        """
        Visualize the path planning solution
        """
        # Generate fallback path in case QUBO solution is not good
        fallback_path = self.path_generator.generate_demo_path()
        
        # Use visualizer to create the plot
        self.visualizer.visualize_solution(solution, fallback_path)
    
    def get_problem_info(self) -> dict:
        """
        Get comprehensive problem information
        """
        size_info = self.quantizer.get_problem_size()
        
        return {
            'parameters': self.problem_params,
            'constraints': {
                'area1_shape': (self.constraints['A1'].shape, self.constraints['b1'].shape),
                'area2_corridors': len(self.constraints['A2_list']),
                'area3_shape': (self.constraints['A3'].shape, self.constraints['b3'].shape)
            },
            'problem_size': size_info
        }
    
    def set_solver_config(self, m_budget: int = None, num_reads: int = None, token: str = None):
        """
        Configure the QDeepSDK solver
        """
        if m_budget is not None:
            self.solver.m_budget = m_budget
        if num_reads is not None:
            self.solver.num_reads = num_reads
        if token is not None:
            self.solver.token = token
    
    def generate_demo_path(self):
        """
        Generate a demonstration path for comparison
        """
        return self.path_generator.generate_demo_path()
    
    def visualize_constraint_violations(self, solution: dict):
        """
        Visualize constraint violation analysis
        """
        if solution and 'solution' in solution and 'constraint_violations' in solution['solution']:
            violations = solution['solution']['constraint_violations']
            self.visualizer.plot_constraint_violations(violations)
        else:
            print("No constraint violation data available")
    
    def visualize_corridor_analysis(self, solution: dict):
        """
        Visualize corridor selection analysis
        """
        if solution and 'solution' in solution and 'corridor_selections' in solution['solution']:
            corridor_selections = solution['solution']['corridor_selections']
            self.visualizer.plot_corridor_selection(corridor_selections)
        else:
            print("No corridor selection data available")
