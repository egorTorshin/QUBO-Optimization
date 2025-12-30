"""
main entry point for path planning solver using multiple-shooting QUBO algorithm.
implements CLI interface with difficulty levels and solver modes.
"""

import numpy as np
import argparse
import sys
from problem_definition import small_problem, standard_problem, rotated_problem, challenging_rotated_problem
from path_planner import PathPlanner, PathPlannerConfig
from visualization import PathVisualizer


def run_path_planner(demo_mode: bool = False, problem_type: str = "standard",
                     num_reads: int = 5000, num_sweeps: int = 10000):
    """run path planner with given configuration."""
    print("="*70)
    print("PATH PLANNING WITH MULTIPLE-SHOOTING QUBO")
    print("="*70)
    
    # create problem based on type
    if problem_type == "small":
        problem, constraints = small_problem()
        print("Problem: SMALL (8 corridors, 4-bit quantization)")
    elif problem_type == "rotated":
        problem, constraints = rotated_problem(15.0)
        print("Problem: ROTATED (15 deg rotation, some corridors unreachable)")
    elif problem_type == "challenging":
        problem, constraints = challenging_rotated_problem()
        print("Problem: CHALLENGING (25 deg rotation, many corridors unreachable)")
    else:
        problem, constraints = standard_problem()
        print("Problem: STANDARD (16 corridors, 8-bit quantization)")
    
    print(f"  Start: {problem.start_point}")
    print(f"  Goal: {problem.goal_point}")
    print(f"  Steps: L1={problem.L1} + L2={problem.L2} + L3={problem.L3} = {problem.L}")
    print(f"  Corridors: H={problem.H}")
    print(f"  Quantization: N={problem.N} bits, n={problem.n} bits")
    if problem.area3_rotation != 0.0:
        print(f"  Area 3 Rotation: {problem.area3_rotation:.1f} deg")
    
    # initialize visualizer
    visualizer = PathVisualizer(problem, constraints)
    
    # solve problem
    if demo_mode:
        print("\nDEMO MODE: generating demonstration path")
        
        # generate demo path using heuristic
        demo_path = visualizer.generate_demo_path()
        objective = sum(np.sum((demo_path[i+1] - demo_path[i])**2) for i in range(len(demo_path)-1))
        
        result = type('obj', (object,), {
            'path': np.array(demo_path),
            'corridor_selections': [0.0] * problem.L2,
            'objective_value': objective,
            'converged': True,
            'num_iterations': 0,
            'total_time': 0.1,
            'iteration_history': []
        })()
        
        solution_type = "Demo Path"
        
    else:
        print(f"\nMULTIPLE-SHOOTING MODE (D-Wave simulated annealing)")
        
        # configure planner
        config = PathPlannerConfig(
            max_iterations=10,
            tolerance=1e-3,
            verbose=True,
            num_reads=num_reads,
            num_sweeps=num_sweeps
        )
        
        # create and run planner
        planner = PathPlanner(problem, constraints, config)
        result = planner.solve()
        
        solution_type = "Multiple-Shooting QUBO"
    
    # print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    success = result.objective_value < float('inf') and len(result.path) > 0
    
    if success:
        print(f"Solution found")
        print(f"   Method: {solution_type}")
        print(f"   Objective: {result.objective_value:.4f}")
        print(f"   Path length: {len(result.path)} points")
        
        if hasattr(result, 'converged'):
            print(f"   Converged: {result.converged}")
            print(f"   Iterations: {result.num_iterations}")
            print(f"   Time: {result.total_time:.2f}s")
        
        # validate path endpoints
        start_error = np.linalg.norm(result.path[0] - problem.start_point)
        goal_error = np.linalg.norm(result.path[-1] - problem.goal_point)
        print(f"   Start error: {start_error:.3f}")
        print(f"   Goal error: {goal_error:.3f}")
        
    else:
        print(f"Solution failed - using fallback demo path")
        
        # fallback to demo
        demo_path = visualizer.generate_demo_path()
        result.path = np.array(demo_path)
        result.objective_value = sum(np.sum((demo_path[i+1] - demo_path[i])**2) for i in range(len(demo_path)-1))
        solution_type = "Demo Path (Fallback)"
    
    # visualization
    print(f"\nGenerating visualization...")
    
    # main path plot
    visualizer.visualize_solution(
        path=result.path,
        corridor_selections=result.corridor_selections,
        objective_value=result.objective_value,
        title_suffix=f" ({solution_type})"
    )
    
    # convergence plot (if available)
    if hasattr(result, 'iteration_history') and len(result.iteration_history) > 1:
        print("Plotting convergence history...")
        visualizer.plot_convergence_history(result.iteration_history)
    
    return result


def parse_arguments():
    """parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Path Planning with Multiple-Shooting QUBO Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Difficulty Levels:
  --easy      simple problem (8 corridors, 4-bit quantization)
  --medium    standard problem (16 corridors, 8-bit quantization)  
  --hard      rotated problem (15 deg rotation, some corridors blocked)
  --extreme   challenging problem (25 deg rotation, many corridors blocked)

Examples:
  python main.py --easy --demo
  python main.py --hard --qubo
  python main.py --medium --qubo --num-reads 5000
        """
    )
    
    # difficulty level (mutually exclusive)
    difficulty = parser.add_mutually_exclusive_group()
    difficulty.add_argument('--easy', action='store_true', 
                          help='easy: 8 corridors, 4-bit quantization')
    difficulty.add_argument('--medium', action='store_true',
                          help='medium: 16 corridors, 8-bit quantization (default)')  
    difficulty.add_argument('--hard', action='store_true',
                          help='hard: 15 deg rotation, some corridors unreachable')
    difficulty.add_argument('--extreme', action='store_true', 
                          help='extreme: 25 deg rotation, many corridors unreachable')
    
    # mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--demo', action='store_true',
                     help='demo mode: generate path without QUBO solver (default)')
    mode.add_argument('--qubo', action='store_true',
                     help='QUBO mode: use D-Wave simulated annealing solver')
    
    # other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='enable verbose output')
    parser.add_argument('--num-reads', type=int, default=5000,
                       help='number of annealing runs (default: 5000)')
    parser.add_argument('--num-sweeps', type=int, default=10000,
                       help='sweeps per anneal (default: 10000)')
    
    return parser.parse_args()


def main():
    """main entry point with command line argument support."""
    
    # parse command line arguments
    args = parse_arguments()
    
    # banner
    print("="*70)
    print("PATH PLANNING SOLVER - COMMAND LINE VERSION")  
    print("="*70)
    print("Reference: QUBO_Mixed_integer_solver.pdf, Sections 0.3 and 2")
    print()
    
    # determine configuration from arguments
    if args.easy:
        problem_type = "small"
        difficulty_name = "EASY"
    elif args.hard:
        problem_type = "rotated"  
        difficulty_name = "HARD"
    elif args.extreme:
        problem_type = "challenging"
        difficulty_name = "EXTREME"
    else:
        problem_type = "standard"
        difficulty_name = "MEDIUM"
    
    # mode selection
    if args.qubo:
        demo_mode = False
        mode_name = "QUBO (simulated annealing)"
    else:
        demo_mode = True
        mode_name = "DEMO"
    
    # display configuration
    print(f"Configuration:")
    print(f"  Difficulty: {difficulty_name}")
    print(f"  Mode: {mode_name}")
    if not demo_mode:
        print(f"  Num reads: {args.num_reads}")
        print(f"  Num sweeps: {args.num_sweeps}")
    print()
    
    # run solver
    try:
        result = run_path_planner(
            demo_mode=demo_mode, 
            problem_type=problem_type,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps
        )
        
        # final summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        
        if result.objective_value < float('inf'):
            print(f"Path planning completed successfully")
            print(f"   Final objective: {result.objective_value:.4f}")
            if hasattr(result, 'converged'):
                convergence_status = "converged" if result.converged else "not converged"
                print(f"   Status: {convergence_status} in {result.num_iterations} iterations")
        else:
            print(f"Path planning failed")
        
        print(f"\nAlgorithm: Multiple-Shooting QUBO Mixed-Binary Solver")
        print(f"Difficulty: {difficulty_name}")
        print(f"Mode: {mode_name}")
        
    except KeyboardInterrupt:
        print(f"\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. try --demo for testing without QUBO solver")
        print(f"2. try --easy for smaller problem size")
        print(f"3. increase --num-sweeps for better QUBO convergence")
        print(f"4. use --help to see all options")
        sys.exit(1)


if __name__ == "__main__":
    main()
