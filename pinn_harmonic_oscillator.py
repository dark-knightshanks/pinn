#Using PyTorch, solve the damped harmonic oscillator [5] using a PINN. Choose fixed initial conditions:
#x(0) = x₀, dx/dz(0) = v₀, with x₀ = 0.7 and v₀ = 1.2.
#Condition the PINN on damping ratios in the range ξ = 0.1 to 0.4.
#Solve on the domain z ∈ [0, 20]:
#d²x/dz² + 2ξ·dx/dz + x = 0

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random as random
import matplotlib as plt
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.optim.lr_scheduler as lr_scheduler

# need z in domain [0,20]
seed = 670
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
n = 2500
z = torch.rand(n,1)*20
damp_rate = torch.rand(n,1)*0.3 + 0.1
n_init = 2500
z_init = torch.zeros(n_init,1)
damp_rate_init = torch.rand(n_init,1)*0.3 + 0.1
x_init = 0.7
v_init = 1.2
learning_rate = 0.0005

from torch.nn.modules.linear import Linear
class pinn(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
       nn.Linear(2,20),
       nn.ReLU(),
       nn.Linear(20,20),
       nn.ReLU(),
       nn.Linear(20,1)
    )
  def forward(self,z,damp_rate):
    inputs = torch.cat([z,damp_rate], dim=1)
    return self.network(inputs)

#loss function d²x/dz² + 2ξ·dx/dz + x  must be close to zero
torch.manual_seed(seed)
model = pinn()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Add cosine annealing scheduler
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=2000,  # Total epochs
    eta_min=1e-6  # Minimum learning rate
)

def physics_loss(model, z, damp_rate):
  z.requires_grad = True
    # Get prediction
  x = model(z, damp_rate)
    # Compute first derivative dx/dz
  x_dot = torch.autograd.grad(x, z,
                                 grad_outputs=torch.ones_like(x),
                                 create_graph=True)[0]

    # Compute second derivative d²x/dz²
  x_ddot = torch.autograd.grad(x_dot, z,
                                   grad_outputs=torch.ones_like(x_dot),
                                   create_graph=True)[0]
  out = x + (2*damp_rate*x_dot) + x_ddot

    # Loss is how far from zero
  return torch.mean(out**2)

def position_loss(model, z_init, damp_rate_init):
  z_init.requires_grad = True
  x_pred = model(z_init, damp_rate_init)
  x_init = 0.7
  v_init = 1.2
  v_pred = torch.autograd.grad(x_pred, z_init,
                               grad_outputs=torch.ones_like(x_pred),
                               create_graph=True)[0]
  return torch.mean((x_init - x_pred)**2) + torch.mean((v_init - v_pred)**2)
losses = []
for epoch in range(2500):
  optimizer.zero_grad()
  loss_physiscs = physics_loss(model,z,damp_rate)
  loss_init = position_loss(model,z_init, damp_rate_init)
  total_loss = loss_physiscs + loss_init
  losses.append(total_loss.item())
  total_loss.backward()
  optimizer.step()
  scheduler.step()
  print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
  # After training with weight = 1 (your best so far)

loss_phys = physics_loss(model, z, damp_rate)
loss_pos = position_loss(model, z_init, damp_rate_init)

print("\n" + "="*50)
print("LOSS BREAKDOWN (weight = 1)")
print("="*50)
print(f"Physics loss:  {loss_phys.item():.8f}")
print(f"Position loss: {loss_pos.item():.8f}")

print(f"Total:{(loss_phys + loss_pos ).item():.8f}")
print("="*50)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel(losses)
plt.xlabel(epoch)
plt.show

import matplotlib.pyplot as plt
import numpy as np

# Create test points
z_test = torch.linspace(0, 20, 400).reshape(-1, 1)
damping_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Multiple damping ratios
ax1 = axes[0]
for xi in damping_values:
    xi_tensor = torch.ones(400, 1) * xi

    with torch.no_grad():
        x_pred = model(z_test, xi_tensor)

    ax1.plot(z_test.numpy(), x_pred.numpy(),
             label=f'ξ = {xi}', linewidth=2, alpha=0.8)

ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.axhline(y=0.7, color='r', linestyle=':', alpha=0.5, linewidth=1, label='x(0) = 0.7')
ax1.set_xlabel('Time (z)', fontsize=12)
ax1.set_ylabel('Position x(z)', fontsize=12)
ax1.set_title('Damped Harmonic Oscillator: PINN Solutions for Different Damping Ratios', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(0, 20)

# Plot 2: Zoom in on first few time units to see initial behavior
ax2 = axes[1]
z_test_zoom = torch.linspace(0, 5, 200).reshape(-1, 1)

for xi in [0.1, 0.25, 0.4]:
    xi_tensor = torch.ones(200, 1) * xi

    with torch.no_grad():
        x_pred = model(z_test_zoom, xi_tensor)

    ax2.plot(z_test_zoom.numpy(), x_pred.numpy(),
             label=f'ξ = {xi}', linewidth=2.5, marker='o', markersize=3, markevery=20)

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=0.7, color='r', linestyle=':', alpha=0.5, label='x(0) = 0.7')
ax2.axvline(x=0, color='g', linestyle=':', alpha=0.5)
ax2.set_xlabel('Time (z)', fontsize=12)
ax2.set_ylabel('Position x(z)', fontsize=12)
ax2.set_title('Zoom: Initial Behavior (first 5 time units)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 5)

plt.tight_layout()
plt.show()

# Verify initial conditions numerically
print("\n" + "="*60)
print("INITIAL CONDITION VERIFICATION")
print("="*60)

for xi in [0.1, 0.2, 0.3, 0.4]:
    z_zero = torch.zeros(1, 1)
    z_zero.requires_grad = True
    xi_tensor = torch.tensor([[xi]])

    x0 = model(z_zero, xi_tensor)
    v0 = torch.autograd.grad(x0, z_zero,
                             grad_outputs=torch.ones_like(x0),
                             create_graph=True)[0]

    x_error = abs(x0.item() - 0.7)
    v_error = abs(v0.item() - 1.2)

    print(f"\nξ = {xi}:")
    print(f"  x(0) = {x0.item():.6f}  (target: 0.700, error: {x_error:.6f})")
    print(f"  v(0) = {v0.item():.6f}  (target: 1.200, error: {v_error:.6f})")

print("="*60)
