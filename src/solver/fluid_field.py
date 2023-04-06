import taichi as ti
import taichi.math as tim
from .diffusion import diffuse_density, diffuse_velocity
from .advection import advect_density, advect_velocity
from .projection import project
from .temporal_value_field import TemporalValueField
from .field_helpers import add_source

@ti.data_oriented
class FluidField:
  def __init__(self, n):
    self.n = n

    self.viscosity = 0
    self.diffusion_rate = 0

    self.boundry_layer = 2
    field_size = n + self.boundry_layer

    self.density = TemporalValueField((field_size, field_size), float)
    self.h_velocity = TemporalValueField((field_size, field_size), float)
    self.v_velocity = TemporalValueField((field_size, field_size), float)
    self.pressure = ti.field(dtype=float, shape=(field_size, field_size))
    self.divergence = ti.field(dtype=float, shape=(field_size, field_size))

  @ti.kernel
  def reset_fields(self):
    for i, j in self.density.previous:
      self.density.previous[i, j] = 0
      self.h_velocity.previous[i, j] = 0
      self.v_velocity.previous[i, j] = 0

  def step(self, dt: float):
    self.velocity_step(dt)
    self.density_step(dt)

  def density_step(self, dt: float):
    add_source(self.density.current, self.density.previous, dt)

    self.density.swap()
    diffuse_density(self.n, dt, self.diffusion_rate, self.density)

    self.density.swap()
    advect_density(self.n, dt, self.density, self.h_velocity, self.v_velocity)

  def velocity_step(self, dt: float):
    add_source(self.h_velocity.current, self.h_velocity.previous, dt)
    add_source(self.v_velocity.current, self.v_velocity.previous, dt)

    self.h_velocity.swap()
    self.v_velocity.swap()
    diffuse_velocity(self.n, dt, self.viscosity, self.h_velocity, self.v_velocity)

    project(self.n, self.h_velocity.current, self.v_velocity.current, self.pressure, self.divergence)

    self.h_velocity.swap()
    self.v_velocity.swap()

    advect_velocity(self.n, dt, self.h_velocity, self.v_velocity, self.h_velocity, self.v_velocity)	

    project(self.n, self.h_velocity.current, self.v_velocity.current, self.pressure, self.divergence)

    
