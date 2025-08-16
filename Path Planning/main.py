import numpy as np
from path_planning_solver import PathPlanningSolver

def run_with_valid_token(token: str):
    """
    Run the path planning problem with a valid QDeepSDK token
    
    Args:
        token: Your valid QDeepSDK authentication token
    """
    print("Path Planning with QDeepSDK - Authenticated Run")
    print("=" * 60)
    
    # Problem parameters matching the screenshot
    start_point = np.array([-3.0, 0.0])  # Start in bottom area, left of center
    goal_point = np.array([-3.0, 10.0])  # Goal in top area, left of center
    L1 = 4  # Steps in area 1 (diagonal movement)
    L2 = 6  # Steps in corridors (vertical movement)
    L3 = 4  # Steps in area 3 (diagonal movement)
    H = 16  # Number of corridors (as shown in screenshot)
    N = 8   # Bits for position quantization
    n = 6   # Bits for slack quantization
    
    # Create path planning solver
    solver = PathPlanningSolver(
        start_point=start_point,
        goal_point=goal_point,
        L1=L1, L2=L2, L3=L3, H=H, N=N, n=n
    )
    
    print(f"Problem setup:")
    print(f"  Start point: {start_point}")
    print(f"  Goal point: {goal_point}")
    print(f"  Total steps: {solver.problem_params['L']} (L1={L1}, L2={L2}, L3={L3})")
    print(f"  Number of corridors: {H}")
    print(f"  Quantization bits: N={N}, n={n}")
    
    # Set authentication token and solver configuration
    print(f"\nSetting up authentication...")
    solver.set_solver_config(
        token=token,
        m_budget=50000,
        num_reads=10000
    )
    print(f"  Token set: {token[:10]}...{token[-10:] if len(token) > 20 else '***'}")
    
    print(f"Solver configuration:")
    print(f"  Measurement budget: {solver.solver.m_budget}")
    print(f"  Number of reads: {solver.solver.num_reads}")
    
    # Get problem information
    problem_info = solver.get_problem_info()
    print(f"\nProblem size information:")
    size_info = problem_info['problem_size']
    print(f"  Position variables: {size_info['position_vars']}")
    print(f"  Slack variables: {size_info['slack_vars']}")
    print(f"  Corridor variables: {size_info['corridor_vars']}")
    print(f"  Total variables: {size_info['total_vars']}")
    
    # Solve with QDeepSDK
    print("\nSolving with QDeepSDK...")
    try:
        solution = solver.solve_with_qdeephybrid()
        
        if solution['status'] == 'solved':
            print(f"Solution found successfully!")
            print(f"  Energy: {solution['energy']}")
            print(f"  Time: {solution['time']} seconds")
            print(f"  Configuration length: {len(solution['configuration'])}")
            
            # Extract path information
            if 'solution' in solution and solution['solution']:
                path_data = solution['solution']
                print(f"  Path objective value: {path_data['objective_value']:.4f}")
                print(f"  Solution quality: {path_data['solution_quality']}")
            
            # Visualize the solution
            print("\nVisualizing solution...")
            solver.visualize_solution(solution)
            
            return solution
            
        else:
            print(f"Error: {solution['message']}")
            return None
            
    except Exception as e:
        print(f"Exception during solving: {e}")
        return None


def main():
    """
    Main function - replace 'your-token-here' with your actual token
    """
    # Replace this with your actual QDeepSDK token
    token = "akwysie03c"
    
    if token == "your-token-here":
        print("Please replace 'your-token-here' with your actual QDeepSDK token")
        print("Example:")
        print("token = 'abc123def456ghi789'")
        return
    
    print("Starting QDeepSDK Path Planning")
    print("=" * 40)
    
    # Run the main problem
    main_solution = run_with_valid_token(token)
    
    # Summary
    print("\n" + "="*40)
    print("Summary")
    print("="*40)
    
    if main_solution and main_solution['status'] == 'solved':
        print(f"Problem: SOLVED (Energy: {main_solution['energy']:.4f})")
    else:
        print(f"Problem: FAILED")
    
    print("Path Planning completed!")

if __name__ == "__main__":
    main()