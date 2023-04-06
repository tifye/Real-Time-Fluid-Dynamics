import taichi as ti
import taichi.math as tim
from solver.fluid_field import FluidField

ti.init(arch=ti.gpu)

n = 100
cell_size = 10
window_width = n * cell_size
window_height = n * cell_size

window_size = (window_width, window_height)
pixels = ti.field(dtype=ti.f32, shape=window_size)

viscosity = 0
diffusion_rate = 0
time_step = 0.1

force = 0.1
source = 5
source_radius = 3

fluid = FluidField(n)
fluid.viscosity = viscosity
fluid.diffusion_rate = diffusion_rate

window = ti.ui.Window("2D Fluid", res=window_size, pos=(50, 50))
canvas = window.get_canvas()

velocity_vector_width = 0.002

vertices = ti.Vector.field(3, dtype=ti.f32, shape=(n*n*2))
colors = ti.Vector.field(3, dtype=ti.f32, shape=(n*n*2))


@ti.kernel
def render():
  for i, j in pixels:
    cell_x = ti.round(i / cell_size, dtype=int)
    cell_y = ti.round(j / cell_size, dtype=int)
    pixels[i,j] = tim.min(100, fluid.density.current[cell_x,cell_y])


def on_click():
  mouse_x, mouse_y = window.get_cursor_pos()
  x = int(mouse_x * n)
  y = int(mouse_y * n)
  add_source(x, y, fluid.density.previous, source)


def on_right_click():
  mouse_x, mouse_y = window.get_cursor_pos()
  x = int(mouse_x * n)
  y = int(mouse_y * n)
  add_source(x, y, fluid.h_velocity.previous, 0)
  add_source(x, y, fluid.v_velocity.previous, force)


@ti.kernel
def add_source(x: int, y: int, field: ti.template(), value: float):
  for i, j in ti.ndrange((-source_radius, source_radius+1), (-source_radius, source_radius+1)):
    if x + i < 0 or \
      x + i >= n or \
      y + j < 0 or \
      y + j >= n:
      continue
    # add in radius around mouse with range as radius
    if i*i + j*j <= source_radius*source_radius:
      field[x+i, y+j] += value


@ti.kernel
def render_velocity():
  for i, j in ti.ndrange(n, n):
    cell_idx = i * n + j

    x = i * cell_size
    y = j * cell_size

    n_x = x / window_width
    n_y = y / window_height

    vh = fluid.h_velocity.current[i, j]
    vv = fluid.v_velocity.current[i, j]
    velocity = ti.Vector([vh, vv, 0])

    line_start = ti.Vector([n_x, n_y, 0])
    line_end = line_start + velocity.normalized() * 0.01

    vertices[cell_idx * 2] = line_start
    vertices[cell_idx * 2 + 1] = line_end

    vel2d = ti.Vector([vh, vv])
    vel_length = vel2d.norm() * 100
    colors[cell_idx * 2] = (vel_length, 0, 1 - vel_length)
    colors[cell_idx * 2 + 1] = (vel_length, 0, 1 - vel_length)

  
def process_events(window: ti.ui.Window):
  if window.is_pressed(ti.ui.LMB):
    on_click()
  
  if window.is_pressed(ti.ui.RMB):
    on_right_click()




while window.running:
  fluid.reset_fields()

  process_events(window)

  fluid.step(time_step)

  render()
  canvas.set_image(pixels)
  
  render_velocity()
  canvas.lines(vertices=vertices, per_vertex_color=colors, width=velocity_vector_width)
  window.show()
