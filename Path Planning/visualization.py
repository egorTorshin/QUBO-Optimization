"""
Path Visualization and Demo Path Generation.

This module handles visualization of path planning solutions and generates
demonstration paths for testing when QUBO solver is not available.

Combined from: visualization.py + path_generation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from typing import List, Optional
from problem_definition import PathPlanningProblem, EnvironmentConstraints


class PathVisualizer:
    """
    Handles path visualization and demo path generation.
    """

    def __init__(self, problem: PathPlanningProblem, constraints: Optional[dict] = None):
        """
        Initialize with problem definition.

        Args:
            problem: Path planning problem definition
            constraints: Environment constraints (optional, for rotated areas)
        """
        self.problem = problem

        # create environment constraints if not provided
        if constraints is None:
            env = EnvironmentConstraints(problem.H, problem.area3_rotation)
            self.constraints = env.get_all_constraints()
            self.env_constraints = env
        else:
            self.constraints = constraints
            self.env_constraints = EnvironmentConstraints(problem.H, problem.area3_rotation)

    def generate_demo_path(self) -> List[np.ndarray]:
        """
        Generate a demonstration path for testing/fallback.

        Creates a realistic path that:
        1. Starts at start_point
        2. Goes through an ACCESSIBLE corridor (considers Area 3 rotation)
        3. Ends at goal_point

        Returns:
            List of [x, y] positions
        """
        path = []

        # get accessible corridors (considers Area 3 rotation)
        accessible_corridors = self.env_constraints.get_accessible_corridors()

        if not accessible_corridors:
            print(" Warning: No accessible corridors found! Path may be impossible.")
            # fallback: direct path (will likely violate constraints)
            for i in range(self.problem.L):
                t = i / max(self.problem.L - 1, 1)
                point = self.problem.start_point + t * (self.problem.goal_point - self.problem.start_point)
                path.append(point)
            return path

        # choose corridor closest to goal among accessible ones
        goal_x = self.problem.goal_point[0]

        if self.problem.H == 8:
            corridor_spacing = 2.0
            start_x = -7.0
        else:
            corridor_spacing = 1.0
            start_x = -7.5

        # get positions of accessible corridors
        accessible_positions = []
        for corridor_idx in accessible_corridors:
            corridor_x = start_x + corridor_idx * corridor_spacing
            accessible_positions.append((corridor_idx, corridor_x))

        # choose the accessible corridor closest to goal
        selected_corridor_idx, selected_corridor_x = min(
            accessible_positions,
            key=lambda item: abs(item[1] - goal_x)
        )

        print(f" Demo path using accessible corridor {selected_corridor_idx} at x={selected_corridor_x:.1f}")
        print(f"   Accessible corridors: {accessible_corridors}")
        print(f"   Total corridors: {self.problem.H}")

        # Area 1: Start to corridor entrance
        for i in range(self.problem.L1):
            t = i / max(self.problem.L1 - 1, 1)
            corridor_entrance = np.array([selected_corridor_x, 2.0])
            point = self.problem.start_point + t * (corridor_entrance - self.problem.start_point)
            path.append(point)

        # Area 2: Through corridor with smooth motion
        for i in range(self.problem.L2):
            t = i / max(self.problem.L2 - 1, 1)
            y_pos = 2.0 + t * 6.0  # from y=2 to y=8

            # add gentle curve within corridor
            corridor_width = 0.3
            lateral_variation = 0.1 * np.sin(t * np.pi)  # small smooth curve
            point = np.array([selected_corridor_x + lateral_variation, y_pos])
            path.append(point)

        # Area 3: Corridor exit to goal
        for i in range(self.problem.L3):
            t = i / max(self.problem.L3 - 1, 1)
            corridor_exit = np.array([selected_corridor_x, 8.0])
            point = corridor_exit + t * (self.problem.goal_point - corridor_exit)
            path.append(point)

        return path

    def visualize_solution(self, path: np.ndarray, corridor_selections: Optional[List[float]] = None,
                          objective_value: float = 0.0, title_suffix: str = ""):
        """
        Visualize path planning solution.

        Args:
            path: Path as array of [x, y] positions
            corridor_selections: Selected corridors (for coloring)
            objective_value: Path objective value
            title_suffix: Additional title text
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # plot environment areas
        self._plot_environment(ax)

        # plot path
        if len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-o',
                   linewidth=3, markersize=6, label='Path', alpha=0.8)

            # add step numbers (every few steps to avoid clutter)
            step_interval = max(1, len(path) // 10)
            for i in range(0, len(path), step_interval):
                ax.annotate(f'{i+1}', (path[i][0], path[i][1]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='darkblue')

        # plot start and goal
        ax.plot(self.problem.start_point[0], self.problem.start_point[1],
               'go', markersize=12, label='Start', zorder=5)
        ax.plot(self.problem.goal_point[0], self.problem.goal_point[1],
               'ro', markersize=12, label='Goal', zorder=5)

        # add area labels
        ax.text(0, 0, 'Area 1\n(Start)', ha='center', va='center',
               fontsize=11, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0, 5, 'Area 2\n(Corridors)', ha='center', va='center',
               fontsize=11, fontweight='bold', color='darkblue',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0, 10, 'Area 3\n(Goal)', ha='center', va='center',
               fontsize=11, fontweight='bold', color='darkred',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

        # title and labels
        title = f'Path Planning Solution{title_suffix}\n'
        title += f'L1={self.problem.L1}, L2={self.problem.L2}, L3={self.problem.L3}, H={self.problem.H}'
        if objective_value > 0:
            title += f', Objective={objective_value:.3f}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # set axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-4, 14)

        plt.tight_layout()
        # non-blocking visualization - save instead of show
        import os
        import time

        # generate unique filename for PATH solution
        timestamp = int(time.time())
        filename = f"path_solution_{timestamp}_{self.problem.area3_rotation:.0f}deg.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Path plot saved: {filename}")
        plt.close()  # close to free memory

    def _plot_environment(self, ax):
        """Plot the environment areas and corridors."""
        # Area 1 (bottom) - Start region
        rect1 = Rectangle((-8, -2), 16, 4, linewidth=2, edgecolor='green',
                         facecolor='lightgreen', alpha=0.3, label='Area 1')
        ax.add_patch(rect1)

        # Area 2 (middle) - Corridors (show accessible vs inaccessible)
        print(" Computing accessible corridors...")
        corridor_width = 0.3
        accessible_corridors = self.env_constraints.get_accessible_corridors()
        print(f" Found {len(accessible_corridors)}/{self.problem.H} accessible corridors: {accessible_corridors}")

        if self.problem.H == 8:
            corridor_spacing = 2.0
            start_x = -7.0
        else:
            corridor_spacing = 1.0
            start_x = -7.5

        for j in range(self.problem.H):
            center_x = start_x + j * corridor_spacing

            # different colors for accessible vs inaccessible corridors
            is_accessible = j in accessible_corridors
            if is_accessible:
                edge_color = 'blue'
                face_color = 'lightblue'
                alpha = 0.6
                label = 'Accessible' if j == (accessible_corridors[0] if accessible_corridors else -1) else ""
            else:
                edge_color = 'red'
                face_color = 'mistyrose'
                alpha = 0.3
                # find first blocked corridor for label
                blocked_corridors = [x for x in range(self.problem.H) if x not in accessible_corridors]
                label = 'Blocked' if blocked_corridors and j == blocked_corridors[0] else ""

            corridor = Rectangle(
                (center_x - corridor_width/2, 2), corridor_width, 6,
                linewidth=1, edgecolor=edge_color, facecolor=face_color, alpha=alpha,
                label=label
            )
            ax.add_patch(corridor)

            # add corridor numbers and accessibility status
            if j % 4 == 0 or j in accessible_corridors[:3]:  # show numbers for key corridors
                text_color = 'darkblue' if j in accessible_corridors else 'darkred'
                symbol = '✓' if j in accessible_corridors else '✗'
                ax.text(center_x, 5, f'{j}\n{symbol}', ha='center', va='center',
                       fontsize=8, color=text_color, fontweight='bold')

        # Area 3 (top) - Goal region (possibly rotated)
        if self.problem.area3_rotation == 0.0:
            # original rectangular Area 3
            rect3 = Rectangle((-8, 8), 16, 4, linewidth=2, edgecolor='red',
                             facecolor='lightcoral', alpha=0.3, label='Area 3')
            ax.add_patch(rect3)
        else:
            # rotated Area 3
            corners = self.env_constraints.get_rotated_area3_corners()
            polygon = Polygon(corners, linewidth=2, edgecolor='red',
                            facecolor='lightcoral', alpha=0.3, label='Area 3')
            ax.add_patch(polygon)

            # add rotation angle annotation
            center = np.mean(corners, axis=0)
            ax.annotate(f'Rotated {self.problem.area3_rotation:.1f}°',
                       xy=center, ha='center', va='center',
                       fontsize=9, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    def plot_convergence_history(self, iteration_history: List[dict]):
        """
        Plot convergence history from multiple-shooting.

        Args:
            iteration_history: List of iteration data with objectives
        """
        if not iteration_history:
            print("No iteration history to plot")
            return

        iterations = [data['iteration'] for data in iteration_history]
        objectives = [data.get('objective', 0) for data in iteration_history]
        energies = [data.get('qubo_energy', 0) for data in iteration_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # objective convergence
        ax1.plot(iterations, objectives, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Path Objective')
        ax1.set_title('Multiple-Shooting Convergence\n(QP Refined Objective)')
        ax1.grid(True, alpha=0.3)

        # QUBO energy
        ax2.plot(iterations, energies, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('QUBO Energy')
        ax2.set_title('QUBO Energy per Iteration')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        # non-blocking visualization - save instead of show
        import os
        import time

        # generate unique filename for CONVERGENCE history
        timestamp = int(time.time())
        filename = f"convergence_{timestamp}_{self.problem.area3_rotation:.0f}deg.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Convergence plot saved: {filename}")
        plt.close()  # close to free memory

    def compare_paths(self, paths_dict: dict):
        """
        Compare multiple paths on the same plot.

        Args:
            paths_dict: Dict of {label: path} for comparison
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # plot environment
        self._plot_environment(ax)

        # plot paths with different colors
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (label, path) in enumerate(paths_dict.items()):
            if len(path) > 0:
                path_array = np.array(path)
                color = colors[i % len(colors)]
                ax.plot(path_array[:, 0], path_array[:, 1],
                       color=color, linewidth=2, marker='o', markersize=4,
                       label=label, alpha=0.7)

        # plot start/goal
        ax.plot(self.problem.start_point[0], self.problem.start_point[1],
               'go', markersize=12, label='Start', zorder=5)
        ax.plot(self.problem.goal_point[0], self.problem.goal_point[1],
               'ro', markersize=12, label='Goal', zorder=5)

        ax.set_title('Path Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-4, 14)

        plt.tight_layout()
        # non-blocking visualization - save instead of show
        import os
        import time

        # generate unique filename for PATH comparison
        timestamp = int(time.time())
        filename = f"comparison_{timestamp}_{self.problem.area3_rotation:.0f}deg.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Comparison plot saved: {filename}")
        plt.close()  # close to free memory
