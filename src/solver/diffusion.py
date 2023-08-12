import taichi as ti
from .temporal_value_field import TemporalValueField
from .field_helpers import get_adjacent, contain, nullify_boundary_flow


def diffuse_density(n: int, dt: float, viscocity: float, density: TemporalValueField):
  diffusion_rate = dt * viscocity * n * n
  for _ in range(20):
    __diffuse_kernel(n, diffusion_rate, density)
    contain(n, density.current)


def diffuse_velocity(n: int, dt: float, viscocity: float, h_velocity: TemporalValueField, v_velocity: TemporalValueField):
  diffusion_rate = dt * viscocity * n
  for _ in range(20):
    __diffuse_kernel(n, diffusion_rate, h_velocity)
    __diffuse_kernel(n, diffusion_rate, v_velocity)
    nullify_boundary_flow(n, h_velocity.current, v_velocity.current)


def diffuse(n: int, dt: float, viscocity: float, field: TemporalValueField):
  diffusion_rate = dt * viscocity * n * n
  for _ in range(20):
    __diffuse_kernel(n, diffusion_rate, field)
  

@ti.kernel
def __diffuse_kernel(n: int, diffusion_rate: float, tv_field: ti.template()):
  # x = 1; x <= n; n++
  for i, j in ti.ndrange((1, n + 1), (1, n + 1)):
    left, right, up, down = get_adjacent(i, j, tv_field.current)

    surrounding_density = left + right + up + down
    absorb_diffusion = diffusion_rate * surrounding_density

    prev_density = tv_field.previous[i, j]
    numerator = prev_density + absorb_diffusion
    denominator = 1 + 4 * diffusion_rate

    tv_field.current[i, j] = numerator / denominator
    