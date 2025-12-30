
# 🧭 Path Planning via QUBO Optimization


Find the shortest path from **start** to **goal** through **vertical corridors**, using quantum-inspired optimization.  
The algorithm combines **binary quantization**, **iterative QUBO solving**, and **QP refinement**.

---

## 🎯 Task Description

- **Region 1**: Obstacle-free starting zone (bottom)  
- **Region 2**: Vertical corridors (middle)  
- **Region 3**: Obstacle-free goal zone (top)

Algorithm key idea: binary quantization of continuous positions and iterative solution of the QUBO model with QP refinement.

---

## 📁 Project Structure

```
Path Planning/
├── main.py
├── problem_definition.py
├── qubo_solver.py
├── path_planner.py
├── visualization.py
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd "Path Planning"
pip install -r requirements.txt
```

### 2. Run Examples

#### 🟢 Easy (demo mode)
```bash
python main.py --easy
```

#### 🟡 Medium (default)
```bash
python main.py --medium
# or simply
python main.py
```

#### 🔴 Hard (with rotation)
```bash
python main.py --hard
```

#### ⚫ Extreme (25° rotation)
```bash
python main.py --extreme
```

### 3. Run with QUBO Solver (D-Wave Simulated Annealing)
```bash
# local solver, no token required
python main.py --hard --qubo

# with custom parameters
python main.py --medium --qubo --num-reads 5000 --num-sweeps 20000
```

### 4. Display Help
```bash
python main.py --help
```

---

## ⚙️ Configuration

### Problem Sizes
| Size | Corridors | Quantization | Use Case |
|------|-----------|--------------|----------|
| Small | 8 | 4-bit | Fast testing |
| Standard | 16 | 8-bit | Full-scale |

### Algorithm Parameters
- **Max iterations**: 10 (multiple-shooting cycles)
- **Tolerance**: 1e-3 (convergence threshold)
- **Quantization**: Adaptive boundary shrinking

---

## 🧮 Algorithm Overview

**Multiple-Shooting QUBO Algorithm**

1. **Quantize** continuous positions \(x_i\) via binary encoding (eq. 12–13)  
2. **Build QUBO** matrix with penalty terms (eq. 14)  
3. **Solve QUBO** to obtain optimal \(x^*\) and corridor selection \(c^*\)  
4. **QP Refinement** with fixed \(c^*\)  
5. **Adaptive Boundary Shrinking** to improve quantization  
6. **Repeat** until QP no longer modifies \(x^*\)

---

## 📊 Key Features

- ✅ Simple modular architecture — only 4 core modules  
- ✅ Works in **demo mode** (no QUBO solver required)  
- ✅ Adaptive quantization with dynamic boundaries  
- ✅ QP refinement using **CVXPY**  
- ✅ Built-in visualization of paths and iterations  
- ✅ Iteration history and convergence metrics

---

## 🔧 Module Details

### `problem_definition.py`
- `PathPlanningProblem`: Defines problem parameters (start, goal, steps, corridors)  
- `EnvironmentConstraints`: Constraint matrices \(A^{(k)}, b^{(k)}\) for 3 regions  
- `create_problem()`: Factory for standard configurations

### `qubo_solver.py`
- `QUBOSolver`: Complete QUBO pipeline  
- Binary variable indexing and quantization (eq. 12–13)  
- Builds QUBO with penalty terms (eq. 14)  
- Decodes solution and validates constraints

### `path_planner.py`
- `PathPlanner`: Core multiple-shooting logic  
- `PathPlannerConfig`: Solver configuration  
- QP refinement via CVXPY or projection  
- Adaptive boundary shrinking and convergence detection

### `visualization.py`
- `PathVisualizer`: Path plots and environment rendering  
- Demo path generation for testing/fallback  
- Convergence history visualization

---

## 🧩 Example Usage

### Basic Example
```python
from problem_definition import standard_problem
from path_planner import PathPlanner, PathPlannerConfig

# Create problem
problem, constraints = standard_problem()

# Configure solver
config = PathPlannerConfig(
    max_iterations=10,
    qubo_token="your-token"
)

# Solve
planner = PathPlanner(problem, constraints, config)
result = planner.solve()

print(f"Objective value: {result.objective_value:.3f}")
print(f"Converged: {result.converged}")
```

### Custom Problem
```python
from problem_definition import create_problem

problem, constraints = create_problem(
    start_point=[-3.0, 0.0],
    goal_point=[-3.0, 10.0],
    L1=4, L2=6, L3=4,  # Steps per region
    H=16,              # Corridors
    N=8, n=6           # Quantization bits
)
```

---

## 📈 Performance

**Scaling:**
| Case | Variables | Parameters |
|------|-----------|------------|
| Small | 332 | L=10, H=8, N=4, n=4 |
| Standard | 1,248 | L=14, H=16, N=8, n=6 |

**Runtime:**
- Demo mode: < 1 sec  
- QUBO solving: 10–60 sec/iteration  
- Multiple-shooting: 2–10 iterations

**Variables formula**: ~L×(2N + 4n) + L2×H binary variables

---

## 📚 References

- Multiple-Shooting QUBO Method (Section 0.3, internal report PDF)
- [CVXPY documentation](https://www.cvxpy.org/)
- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)

---

## 👨‍💻 Author

**Quantum Optimization Research — QDeep** • 2025  
**Maintainer**: [Your GitHub handle]

---

*Ready for production use in quantum path planning research.*
