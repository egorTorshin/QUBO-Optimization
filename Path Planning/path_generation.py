import numpy as np
from typing import List

class PathGenerator:
    """
    Handles generation of demonstration/fallback paths
    """
    
    def __init__(self, problem_params: dict):
        """
        Initialize path generator with problem parameters
        
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
    
    def generate_demo_path(self) -> List[np.ndarray]:
        """
        Generate a demonstration path with realistic positioning
        """
        path = []
        
        # Use the actual start and goal points from the problem
        start_point = self.start_point
        goal_point = self.goal_point
        
        # Select a corridor that makes sense for the path
        # Choose a corridor that's between start and goal
        start_x = start_point[0]
        goal_x = goal_point[0]
        target_x = (start_x + goal_x) / 2  # Middle corridor
        
        # Find the closest corridor (adjust for problem size)
        if self.H == 8:
            # Smaller problem with 8 corridors: wider spacing
            corridor_positions = [-7.0 + i * 2.0 for i in range(8)]
        else:
            # Main problem with 16 corridors: original spacing
            corridor_positions = [-7.5 + i * 1.0 for i in range(16)]
        selected_corridor_x = min(corridor_positions, key=lambda x: abs(x - target_x))
        
        # Segment 1: Start to corridor entrance (smooth diagonal)
        for i in range(self.L1):
            t = i / max(self.L1 - 1, 1)
            corridor_entrance = np.array([selected_corridor_x, 2.0])
            # Smooth interpolation from start to corridor entrance
            point = start_point + t * (corridor_entrance - start_point)
            path.append(point)
        
        # Segment 2: Through selected corridor (vertical with small lateral movement)
        for i in range(self.L2):
            t = i / max(self.L2 - 1, 1)
            y_pos = 2.0 + t * 6.0  # From y=2 to y=8
            
            # Add small realistic lateral movement within corridor
            corridor_width = 0.3
            max_lateral = corridor_width / 2 - 0.05
            lateral_variation = max_lateral * 0.5 * np.sin(t * np.pi)  # Gentle curve
            point = np.array([selected_corridor_x + lateral_variation, y_pos])
            path.append(point)
        
        # Segment 3: From corridor exit to goal (smooth diagonal)
        for i in range(self.L3):
            t = i / max(self.L3 - 1, 1)
            corridor_exit = np.array([selected_corridor_x, 8.0])
            # Smooth interpolation from corridor exit to goal
            point = corridor_exit + t * (goal_point - corridor_exit)
            path.append(point)
        
        return path
    
    def generate_straight_line_path(self) -> List[np.ndarray]:
        """
        Generate a simple straight-line path from start to goal
        """
        path = []
        total_steps = self.L1 + self.L2 + self.L3
        
        for i in range(total_steps):
            t = i / max(total_steps - 1, 1)
            point = self.start_point + t * (self.goal_point - self.start_point)
            path.append(point)
        
        return path
    
    def generate_corridor_specific_path(self, corridor_x: float) -> List[np.ndarray]:
        """
        Generate a path through a specific corridor
        
        Args:
            corridor_x: X-coordinate of the corridor to use
        """
        path = []
        
        # Segment 1: Start to corridor entrance
        for i in range(self.L1):
            t = i / max(self.L1 - 1, 1)
            corridor_entrance = np.array([corridor_x, 2.0])
            point = self.start_point + t * (corridor_entrance - self.start_point)
            path.append(point)
        
        # Segment 2: Through specified corridor
        for i in range(self.L2):
            t = i / max(self.L2 - 1, 1)
            y_pos = 2.0 + t * 6.0  # From y=2 to y=8
            point = np.array([corridor_x, y_pos])
            path.append(point)
        
        # Segment 3: From corridor exit to goal
        for i in range(self.L3):
            t = i / max(self.L3 - 1, 1)
            corridor_exit = np.array([corridor_x, 8.0])
            point = corridor_exit + t * (self.goal_point - corridor_exit)
            path.append(point)
        
        return path
