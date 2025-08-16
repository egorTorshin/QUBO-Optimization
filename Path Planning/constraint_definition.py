import numpy as np
from typing import List, Tuple

class ConstraintDefinition:
    """
    Handles definition of constraints for different areas in the path planning problem
    """
    
    def __init__(self, H: int):
        """
        Initialize constraint definition with number of corridors
        
        Args:
            H: Number of corridors in area 2
        """
        self.H = H
    
    def define_area1_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define constraints for obstacle-free area 1 (bottom area)
        Returns: A1, b1 where A1*x <= b1 defines the feasible region
        """
        # Bottom rectangular area as shown in the screenshot
        # Wide horizontal rectangle at the bottom
        A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # x_min, x_max, y_min, y_max
        b1 = np.array([8, -8, 2, -2])  # Wide rectangle: x in [-8,8], y in [-2,2]
        return A1, b1
    
    def define_area2_constraints(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Define constraints for obstacle-free corridors in area 2 (middle section)
        Returns: List of (A2_j, b2_j) for each corridor j
        """
        # Middle section with vertical corridors as shown in the screenshot
        A2_list = []
        b2_list = []
        
        # Create vertical corridors (narrow rectangles)
        corridor_width = 0.3  # Narrow corridors
        
        # Use actual H parameter for number of corridors
        num_corridors = self.H
        
        # Adjust corridor positioning based on number of corridors
        if num_corridors == 8:
            # Smaller problem: wider spacing
            corridor_spacing = 2.0
            start_x = -7.0
        else:
            # Main problem: original spacing
            corridor_spacing = 1.0
            start_x = -7.5
        
        for j in range(num_corridors):
            # Calculate corridor center position
            center_x = start_x + j * corridor_spacing
            
            # Vertical corridor constraints: narrow rectangle
            A2_j = np.array([
                [1, 0],   # x <= center_x + width/2
                [-1, 0],  # -x <= -(center_x - width/2)
                [0, 1],   # y <= 8 (top of corridor area)
                [0, -1]   # -y <= -2 (bottom of corridor area)
            ])
            b2_j = np.array([
                center_x + corridor_width/2,  # Right boundary
                -(center_x - corridor_width/2),  # Left boundary
                8,  # Top boundary
                -2   # Bottom boundary
            ])
            
            A2_list.append(A2_j)
            b2_list.append(b2_j)
        
        return A2_list, b2_list
    
    def define_area3_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define constraints for obstacle-free area 3 (top area)
        Returns: A3, b3 where A3*x <= b3 defines the feasible region
        """
        # Top rectangular area as shown in the screenshot
        # Wide horizontal rectangle at the top, slightly tilted
        # For simplicity, using rectangular constraints (can be extended to tilted)
        A3 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b3 = np.array([8, -8, 12, 8])  # Wide rectangle: x in [-8,8], y in [8,12]
        return A3, b3
    
    def get_all_constraints(self) -> dict:
        """
        Get all area constraints as a dictionary
        """
        A1, b1 = self.define_area1_constraints()
        A2_list, b2_list = self.define_area2_constraints()
        A3, b3 = self.define_area3_constraints()
        
        return {
            'A1': A1, 'b1': b1,
            'A2_list': A2_list, 'b2_list': b2_list,
            'A3': A3, 'b3': b3
        }
