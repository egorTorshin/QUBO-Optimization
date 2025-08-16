import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional

class PathVisualization:
    """
    Handles visualization of path planning solutions
    """
    
    def __init__(self, problem_params: dict):
        """
        Initialize visualization with problem parameters
        
        Args:
            problem_params: Dictionary of problem parameters
        """
        self.params = problem_params
        self.start_point = problem_params['start_point']
        self.goal_point = problem_params['goal_point']
        self.L1 = problem_params['L1']
        self.L2 = problem_params['L2']
        self.L3 = problem_params['L3']
        self.H = problem_params['H']
        self.N = problem_params['N']
        self.n = problem_params['n']
    
    def visualize_solution(self, solution: dict, fallback_path: Optional[List[np.ndarray]] = None):
        """
        Visualize the path planning solution
        """
        print("Generating visualization...")
        
        # Extract path from solution
        if solution and solution.get('status') == 'solved' and 'solution' in solution:
            # Use the actual QUBO solution
            path_data = solution['solution']
            path = path_data['path']
            
            # Check if QUBO solution is reasonable
            if 'constraint_violations' in path_data:
                violations = path_data['constraint_violations']
                start_error = violations['boundary_violations'][0]
                goal_error = violations['boundary_violations'][1]
                
                # If errors are too large, use fallback path
                # More lenient threshold for smaller problems
                error_threshold = 3.0 if self.params['L'] <= 10 else 5.0
                if start_error > error_threshold or goal_error > error_threshold:
                    print(f"QUBO solution has high errors (start: {start_error:.2f}, goal: {goal_error:.2f}), using demo path")
                    path = fallback_path
                else:
                    print(f"Using QUBO solution with objective value: {path_data['objective_value']:.4f}")
            else:
                print(f"Using QUBO solution with objective value: {path_data['objective_value']:.4f}")
        else:
            # Fall back to demo path if no valid solution
            print("No valid QUBO solution, using demo path")
            path = fallback_path
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot areas
        self._plot_areas(ax)
        
        # Plot path
        if path is not None:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-o', linewidth=3, markersize=8, label='Planned Path')
            
            # Add step numbers
            for i, point in enumerate(path):
                ax.annotate(f'{i+1}', (point[0], point[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Plot start and goal
        ax.plot(self.start_point[0], self.start_point[1], 'go', markersize=15, label='Start')
        ax.plot(self.goal_point[0], self.goal_point[1], 'ro', markersize=15, label='Goal')
        
        # Add area labels matching the screenshot
        ax.text(-4, 0, 'start', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='darkgreen')
        ax.text(0, 5, 'Corridors', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkblue')
        ax.text(-4, 10, 'goal', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='darkred')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Path Planning Solution with QDeepHybridSolver\n' + 
                    f'L1={self.L1}, L2={self.L2}, L3={self.L3}, H={self.H}, N={self.N}, n={self.n}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set axis limits to match the screenshot layout
        ax.set_xlim(-10, 10)
        ax.set_ylim(-4, 14)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_areas(self, ax):
        """
        Plot the different areas and corridors exactly as shown in the screenshot
        """
        # Plot Area 1 (bottom area - start region)
        rect1 = Rectangle((-8, -2), 16, 4, linewidth=2, edgecolor='green', 
                         facecolor='lightgreen', alpha=0.3, label='Area 1 (Start)')
        ax.add_patch(rect1)
        
        # Plot Area 2 (middle section - vertical corridors)
        # Use problem-specific number of corridors
        corridor_width = 0.3
        corridor_spacing = 1.0
        
        # For smaller problems (H=8), use fewer corridors
        if hasattr(self, 'H'):
            num_corridors = self.H
        else:
            num_corridors = 16  # Default for main problem
            
        # Adjust corridor spacing for smaller problems
        if num_corridors == 8:
            corridor_spacing = 2.0  # Wider spacing for 8 corridors
            start_x = -7.0  # Start position for 8 corridors
        else:
            corridor_spacing = 1.0  # Original spacing for 16 corridors
            start_x = -7.5  # Original start position
        
        for j in range(num_corridors):
            center_x = start_x + j * corridor_spacing
            corridor = Rectangle((center_x - corridor_width/2, -2), 
                               corridor_width, 10, linewidth=1, edgecolor='blue', 
                               facecolor='lightblue', alpha=0.3)
            ax.add_patch(corridor)
        
        # Plot Area 3 (top area - goal region)
        rect3 = Rectangle((-8, 8), 16, 4, linewidth=2, edgecolor='red', 
                         facecolor='lightcoral', alpha=0.3, label='Area 3 (Goal)')
        ax.add_patch(rect3)
    
    def plot_constraint_violations(self, violations: dict):
        """
        Plot constraint violation analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Boundary violations
        boundary_errors = violations['boundary_violations']
        ax1.bar(['Start Error', 'Goal Error'], boundary_errors)
        ax1.set_title('Boundary Condition Violations')
        ax1.set_ylabel('Error Magnitude')
        
        # Area violations
        area_names = ['Area 1', 'Area 2', 'Area 3']
        area_violations = [
            len(violations['area1_violations']),
            len(violations['area2_violations']),
            len(violations['area3_violations'])
        ]
        ax2.bar(area_names, area_violations)
        ax2.set_title('Number of Area Constraint Violations')
        ax2.set_ylabel('Number of Violations')
        
        # Area 1 violation distribution
        if violations['area1_violations']:
            ax3.hist(violations['area1_violations'], bins=20, alpha=0.7)
            ax3.set_title('Area 1 Violation Distribution')
            ax3.set_xlabel('Violation Magnitude')
            ax3.set_ylabel('Frequency')
        
        # Area 3 violation distribution
        if violations['area3_violations']:
            ax4.hist(violations['area3_violations'], bins=20, alpha=0.7)
            ax4.set_title('Area 3 Violation Distribution')
            ax4.set_xlabel('Violation Magnitude')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def plot_corridor_selection(self, corridor_selections: List[float]):
        """
        Plot corridor selection analysis
        """
        if not corridor_selections:
            print("No corridor selections to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Corridor selection over steps
        ax1.plot(range(len(corridor_selections)), corridor_selections, 'bo-')
        ax1.set_title('Corridor Selection Over Steps')
        ax1.set_xlabel('Corridor Step')
        ax1.set_ylabel('Selected Corridor X-Position')
        ax1.grid(True, alpha=0.3)
        
        # Corridor frequency
        unique_corridors, counts = np.unique(corridor_selections, return_counts=True)
        ax2.bar(unique_corridors, counts)
        ax2.set_title('Corridor Selection Frequency')
        ax2.set_xlabel('Corridor X-Position')
        ax2.set_ylabel('Selection Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
