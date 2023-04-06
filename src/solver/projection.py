import taichi as ti
from .field_helpers import get_adjacent, contain, nullify_boundary_flow

def project(n: int, 
    h_velocity: ti.template(), 
    v_velocity: ti.template(), 
    pressure: ti.template(), 
    divergence: ti.template()
  ):
  h = 1.0 / n
  
  __init_divergence_and_pressure(n, h, h_velocity, v_velocity, pressure, divergence)
  contain(n, divergence)
  contain(n, pressure)

  for _ in range(20):
    __develop_pressure(n, pressure, divergence)
    contain(n, pressure)

  __project_kernel(n, h, h_velocity, v_velocity, pressure)
  nullify_boundary_flow(n, h_velocity, v_velocity)


@ti.kernel
def __init_divergence_and_pressure(n: int, h: float, h_velocity: ti.template(), v_velocity: ti.template(), pressure: ti.template(), divergence: ti.template()):
  for i, j in ti.ndrange((1, n + 1), (1, n + 1)):
    left, right, _, _ = get_adjacent(i, j, h_velocity)
    _, _, up, down = get_adjacent(i, j, v_velocity)

    divergence[i, j] = -0.5 * h * (right - left + up - down)
    pressure[i, j] = 0


@ti.kernel
def __develop_pressure(n: int, pressure: ti.template(), divergence: ti.template()):
  for i, j in ti.ndrange((1, n + 1), (1, n + 1)):
    left, right, up, down = get_adjacent(i, j, pressure)
    pressure[i, j] = (divergence[i, j] + left + right + up + down) / 4


@ti.kernel
def __project_kernel(n: int, h: float, h_velocity: ti.template(), v_velocity: ti.template(), pressure: ti.template()):
  for i, j in ti.ndrange((1, n + 1), (1, n + 1)):
    left, right, up, down = get_adjacent(i, j, pressure)

    h_velocity[i, j] -= 0.5 * (right - left) / h
    v_velocity[i, j] -= 0.5 * (up - down) / h
  

