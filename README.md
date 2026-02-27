# Physics-Informed Neural Networks for Damped Harmonic Oscillator

Implementation of a PINN to solve the damped harmonic oscillator differential equation.

## Problem Statement
Solve the second-order ODE:

```
d²x/dz² + 2ξ·dx/dz + x = 0
```

where:

- `x(z)` is position at time `z`
- `ξ` is the damping ratio (0.1 to 0.4)
- Initial conditions: `x(0) = 0.7`, `dx/dz(0) = 1.2`
- Domain: `z ∈ [0, 20]`

## Implementation

### Network Architecture

- Input: 2D (time z, damping ratio ξ)
- Hidden: 2 layers × 50 neurons with RELU activation
- Output: 1D (position x)

### Loss Function

Three components with equal weighting:

1. Physics loss: Differential equation residual
2. Position loss: `x(0) = 0.7`
3. Velocity loss: `dx/dz(0) = 1.2`

### Training Configuration

- Optimizer: Adam (lr=0.0005)
- Scheduler: Cosine Annealing
- Collocation points: 2500
- Epochs: 2500

## Results

- Final loss: 0.07
- Physics loss: 0.031
- Initial condition loss: 0.039
- Initial condition accuracy: <5% error

The solution correctly reproduces:

- Oscillatory motion with exponential decay
- Damping-dependent behaviour (more damping → faster decay)
- Proper initial conditions
## Repository Structure

```
.
├── README.md
├── pinn_harmonic_oscillator.py    # Main implementation
├── results/
│   ├── loss_curve.png
│   └── solution_plot.png
```

## Key Challenges Solved

1. **Gradient accumulation**: Added `optimizer.zero_grad()` at start of each epoch
2. **Loss balancing**: Found equal weights work best (avoiding over-emphasis on ICs)
3. **Learning rate tuning**: lr=0.0005 with scheduler provides stable convergence
4. **Reproducibility**: Proper seed management for consistent results
