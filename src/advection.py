import taichi as ti
import taichi.math as tim
from field_helpers import contain, bilinear_interpolate_nearest, nullify_boundary_flow

def advect_density(n, dt, density, h_velocity, v_velocity):
  advect_kernel(n, dt, density, h_velocity, v_velocity)
  contain(n, density.current)


def advect_velocity(n, dt, h_velocity, v_velocity, h_velocity_prev, v_velocity_prev):
  advect_kernel(n, dt, h_velocity, h_velocity_prev, v_velocity_prev)
  advect_kernel(n, dt, v_velocity, h_velocity_prev, v_velocity_prev)
  nullify_boundary_flow(n, h_velocity.current, v_velocity.current)


@ti.kernel
def advect_kernel(n: int, dt: float, tv_field: ti.template(), tv_h_velocity: ti.template(), tv_v_velocity: ti.template()):
  n_scale = dt * n
  for i, j in ti.ndrange((1, n + 1), (1, n + 1)):
    # P is the particle at (i, j)
    # Move P back in time by dt 
    # to get the position of the particle at the start of the time step
    x = i - n_scale * tv_h_velocity.current[i, j]
    y = j - n_scale * tv_v_velocity.current[i, j]

    # Clamp P_start to the grid 
    # with a 0.5 unit border
    x = tim.max(0.5, tim.min(n + 0.5, x))
    y = tim.max(0.5, tim.min(n + 0.5, y))

    tv_field.current[i, j] = bilinear_interpolate_nearest(x, y, tv_field.previous)
