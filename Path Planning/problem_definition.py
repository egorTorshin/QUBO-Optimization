"""
Path Planning Problem Definition.

This module defines the path planning problem from QUBO_Mixed_integer_solver.pdf (Section 2):
- Problem parameters (start, goal, steps L1/L2/L3, corridors H)
- Constraint matrices A^(1), A^(2)_j, A^(3) and vectors b^(1), b^(2)_j, b^(3)
- Area definitions for the environment
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PathPlanningProblem:
    """
    Complete path planning problem definition.

    Attributes:
        start_point: Starting position [x, y]
        goal_point: Goal position [x, y]
        L1, L2, L3: Steps in Areas 1, 2, 3
        H: Number of corridors
        N, n: Quantization bits
        sigma_x, delta_x: Position quantization parameters
        sigma_s, delta_s: Slack quantization parameters
        M: Big-M parameter for constraints
        area3_rotation: Rotation angle of Area 3 in degrees
    """
    start_point: np.ndarray
    goal_point: np.ndarray
    L1: int  # steps in Area 1
    L2: int  # steps in Area 2 (corridors)
    L3: int  # steps in Area 3
    H: int   # number of corridors
    N: int   # position quantization bits
    n: int   # slack quantization bits

    # geometry parameters
    area3_rotation: float = 0.0  # rotation angle in degrees for Area 3

    # quantization parameters (will be computed)
    sigma_x: float = 0.1
    delta_x: float = -10.0
    sigma_s: float = 0.1
    delta_s: float = 0.0
    M: float = 10.0  # big-M parameter

    def __post_init__(self):
        """Compute derived parameters."""
        self.L = self.L1 + self.L2 + self.L3  # total steps
        self.start_point = np.array(self.start_point)
        self.goal_point = np.array(self.goal_point)

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            'start_point': self.start_point,
            'goal_point': self.goal_point,
            'L1': self.L1, 'L2': self.L2, 'L3': self.L3, 'L': self.L,
            'H': self.H, 'N': self.N, 'n': self.n,
            'area3_rotation': self.area3_rotation,
            'sigma_x': self.sigma_x, 'delta_x': self.delta_x,
            'sigma_s': self.sigma_s, 'delta_s': self.delta_s,
            'M': self.M
        }


class EnvironmentConstraints:
    """
    Defines constraint matrices A^(k) and vectors b^(k) for the three areas.

    Areas:
    - Area 1: Obstacle-free start region (bottom)
    - Area 2: Vertical corridors (middle)
    - Area 3: Obstacle-free goal region (top, can be rotated)
    """

    def __init__(self, H: int, area3_rotation: float = 0.0):
        """
        Initialize with number of corridors and Area 3 rotation.

        Args:
            H: Number of corridors
            area3_rotation: Rotation angle for Area 3 in degrees
        """
        self.H = H
        self.area3_rotation = area3_rotation

    def get_area1_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Area 1: Bottom rectangular region for start.
        Returns A1, b1 where A1*x ≤ b1.
        """
        # Rectangle: x ∈ [-8, 8], y ∈ [-2, 2]
        A1 = np.array([
            [1, 0],   # x ≤ 8
            [-1, 0],  # -x ≤ 8  (x ≥ -8)
            [0, 1],   # y ≤ 2
            [0, -1]   # -y ≤ 2  (y ≥ -2)
        ])
        b1 = np.array([8, 8, 2, 2])
        return A1, b1

    def get_area2_constraints(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Area 2: Vertical corridors in middle region.
        Returns list of (A2_j, b2_j) for each corridor j.
        """
        A2_list = []
        b2_list = []

        corridor_width = 0.3

        # Corridor positions based on H
        if self.H == 8:
            corridor_spacing = 2.0
            start_x = -7.0
        else:  # h == 16 or other
            corridor_spacing = 1.0
            start_x = -7.5

        for j in range(self.H):
            center_x = start_x + j * corridor_spacing

            # Corridor j constraints
            A2_j = np.array([
                [1, 0],   # x ≤ center_x + width/2
                [-1, 0],  # -x ≤ -(center_x - width/2)
                [0, 1],   # y ≤ 8
                [0, -1]   # -y ≤ -2
            ])
            b2_j = np.array([
                center_x + corridor_width/2,
                -(center_x - corridor_width/2),
                8,
                -2
            ])

            A2_list.append(A2_j)
            b2_list.append(b2_j)

        return A2_list, b2_list

    def get_area3_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Area 3: Top rectangular region for goal (can be rotated).
        Returns A3, b3 where A3*x ≤ b3.
        """
        if self.area3_rotation == 0.0:
            # no rotation - original rectangular constraints
            # Rectangle: x ∈ [-8, 8], y ∈ [8, 12]
            A3 = np.array([
                [1, 0],   # x ≤ 8
                [-1, 0],  # -x ≤ 8  (x ≥ -8)
                [0, 1],   # y ≤ 12
                [0, -1]   # -y ≤ -8  (y ≥ 8)
            ])
            b3 = np.array([8, 8, 12, -8])
            return A3, b3

        # rotated rectangle
        return self._get_rotated_area3_constraints()

    def _get_rotated_area3_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create constraints for rotated Area 3.
        """
        # original rectangle corners: x ∈ [-8, 8], y ∈ [8, 12]
        corners = np.array([
            [-8, 8],   # bottom-left
            [8, 8],    # bottom-right
            [8, 12],   # top-right
            [-8, 12]   # top-left
        ])

        # center of rotation
        center = np.array([0.0, 10.0])  # center of original rectangle

        # rotation angle in radians
        angle_rad = np.deg2rad(self.area3_rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # rotation matrix
        R = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # rotate corners around center
        rotated_corners = []
        for corner in corners:
            # translate to origin
            translated = corner - center
            # rotate
            rotated = R @ translated
            # translate back
            final_corner = rotated + center
            rotated_corners.append(final_corner)

        rotated_corners = np.array(rotated_corners)

        # create linear constraints from rotated rectangle
        # each edge becomes a constraint a*x + b*y ≤ c
        A3_list = []
        b3_list = []

        # Calculate centroid for determining outward direction
        centroid = np.mean(rotated_corners, axis=0)
        
        n_corners = len(rotated_corners)
        for i in range(n_corners):
            p1 = rotated_corners[i]
            p2 = rotated_corners[(i + 1) % n_corners]

            # edge vector
            edge = p2 - p1
            # normal vector (perpendicular to edge)
            normal = np.array([-edge[1], edge[0]])

            # normalize
            if np.linalg.norm(normal) > 1e-10:
                normal = normal / np.linalg.norm(normal)
                
                # Ensure normal points OUTWARD (away from centroid)
                # This is needed for constraint: normal @ x <= normal @ p1
                edge_midpoint = (p1 + p2) / 2
                to_centroid = centroid - edge_midpoint
                if np.dot(normal, to_centroid) > 0:
                    normal = -normal  # Flip to point outward

                # constraint: normal · x ≤ normal · p1
                # Points inside satisfy this when normal points outward
                A3_list.append(normal)
                b3_list.append(np.dot(normal, p1))

        A3 = np.array(A3_list)
        b3 = np.array(b3_list)

        return A3, b3

    def get_rotated_area3_corners(self) -> np.ndarray:
        """
        Get the corners of rotated Area 3 for visualization.
        Returns array of corner points.
        """
        if self.area3_rotation == 0.0:
            return np.array([
                [-8, 8], [8, 8], [8, 12], [-8, 12]
            ])

        # same rotation logic as in constraints
        corners = np.array([
            [-8, 8], [8, 8], [8, 12], [-8, 12]
        ])

        center = np.array([0.0, 10.0])
        angle_rad = np.deg2rad(self.area3_rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_corners = []
        for corner in corners:
            translated = corner - center
            rotated = R @ translated
            final_corner = rotated + center
            rotated_corners.append(final_corner)

        return np.array(rotated_corners)

    def get_accessible_corridors(self) -> List[int]:
        """
        Determine which corridors can actually reach the rotated Area 3.

        Returns:
            List of corridor indices that are accessible
        """
        if self.area3_rotation == 0.0:
            # no rotation - all corridors accessible
            return list(range(self.H))

        accessible = []

        # get Area 3 corners
        area3_corners = self.get_rotated_area3_corners()

        # find the bottom edge of Area 3 (y-coordinate range at y=8 level)
        area3_bottom_y = 8.0  # where corridors end

        # for each corner, find x-range at y=8
        bottom_intersections = []

        # check each edge of the rotated Area 3
        n_corners = len(area3_corners)
        for i in range(n_corners):
            p1 = area3_corners[i]
            p2 = area3_corners[(i + 1) % n_corners]

            # check if edge crosses y = 8
            if (p1[1] <= area3_bottom_y <= p2[1]) or (p2[1] <= area3_bottom_y <= p1[1]):
                if abs(p2[1] - p1[1]) > 1e-10:  # avoid division by zero
                    # linear interpolation to find x at y = 8
                    t = (area3_bottom_y - p1[1]) / (p2[1] - p1[1])
                    x_intersect = p1[0] + t * (p2[0] - p1[0])
                    bottom_intersections.append(x_intersect)

        if len(bottom_intersections) >= 2:
            # Area 3 accessible x-range at y=8
            x_min = min(bottom_intersections)
            x_max = max(bottom_intersections)

            # check each corridor
            corridor_width = 0.3
            if self.H == 8:
                corridor_spacing = 2.0
                start_x = -7.0
            else:
                corridor_spacing = 1.0
                start_x = -7.5

            for j in range(self.H):
                corridor_center = start_x + j * corridor_spacing
                corridor_left = corridor_center - corridor_width/2
                corridor_right = corridor_center + corridor_width/2

                # check if corridor overlaps with accessible Area 3 range
                if corridor_right >= x_min and corridor_left <= x_max:
                    accessible.append(j)

        return accessible

    def get_all_constraints(self) -> dict:
        """Get all constraint matrices as dictionary."""
        A1, b1 = self.get_area1_constraints()
        A2_list, b2_list = self.get_area2_constraints()
        A3, b3 = self.get_area3_constraints()

        return {
            'A1': A1, 'b1': b1,
            'A2_list': A2_list, 'b2_list': b2_list,
            'A3': A3, 'b3': b3
        }


def create_problem(start_point: List[float], goal_point: List[float],
                   L1: int = 4, L2: int = 6, L3: int = 4, H: int = 16,
                   N: int = 8, n: int = 6, area3_rotation: float = 0.0) -> Tuple[PathPlanningProblem, dict]:
    """
    Create a complete path planning problem.

    Args:
        start_point: [x, y] starting position
        goal_point: [x, y] goal position
        L1, L2, L3: Steps in each area
        H: Number of corridors
        N, n: Quantization bits
        area3_rotation: Rotation angle for Area 3 in degrees

    Returns:
        (problem, constraints): Problem definition and constraint matrices
    """
    problem = PathPlanningProblem(
        start_point=start_point,
        goal_point=goal_point,
        L1=L1, L2=L2, L3=L3, H=H, N=N, n=n,
        area3_rotation=area3_rotation
    )

    env = EnvironmentConstraints(H, area3_rotation)
    constraints = env.get_all_constraints()

    return problem, constraints


# predefined problem configurations
def small_problem() -> Tuple[PathPlanningProblem, dict]:
    """Small test problem: 8 corridors, fewer bits."""
    return create_problem(
        start_point=[-3.0, 0.0],
        goal_point=[5.0, 10.0],
        L1=3, L2=4, L3=3, H=8, N=4, n=4
    )

def standard_problem() -> Tuple[PathPlanningProblem, dict]:
    """Standard problem: matches PDF example."""
    return create_problem(
        start_point=[-3.0, 0.0],
        goal_point=[-3.0, 10.0],
        L1=4, L2=6, L3=4, H=16, N=8, n=6
    )

def rotated_problem(rotation_degrees: float = 15.0) -> Tuple[PathPlanningProblem, dict]:
    """Problem with rotated Area 3: some corridors become unreachable."""
    # For 15° rotation, Area 3 tilts left. 
    # Goal must be inside the rotated Area 3 AND reachable from accessible corridors.
    # Accessible corridors are on the left side (x < 0), so goal should also be on left side.
    # Start should also be on left side for a feasible path.
    return create_problem(
        start_point=[5.0, 0.0],  # left side, reachable from left corridors
        goal_point=[5.0, 10.0],  # inside rotated Area 3
        L1=4, L2=6, L3=4, H=16, N=8, n=6,
        area3_rotation=rotation_degrees
    )

def challenging_rotated_problem() -> Tuple[PathPlanningProblem, dict]:
    """Challenging problem: 25° rotation makes many corridors unreachable."""
    return create_problem(
        start_point=[-3.0, 0.0],
        goal_point=[-1.0, 11.0],  # goal positioned for 25° rotation
        L1=4, L2=6, L3=4, H=16, N=8, n=6,
        area3_rotation=25.0
    )
