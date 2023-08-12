import taichi as ti
import taichi.math as tim


@ti.func
def get_adjacent(i, j, source):
  left = source[i - 1, j]
  right = source[i + 1, j]
  up = source[i, j + 1]
  down = source[i, j - 1]

  return (left, right, up, down)


@ti.kernel
def add_source(target: ti.template(), source: ti.template(), dt: float):
  for i, j in target:
    target[i, j] += dt * source[i, j]
    source[i, j] = 0


@ti.kernel
def contain(n: int, field: ti.template()):
  for i in ti.ndrange((1, n + 1)):
    """ # wrap
    field[0, i] = field[n, i]
    field[n + 1, i] = field[1, i]
    field[i, 0] = field[i, n]
    field[i, n + 1] = field[i, 1] """

    field[0, i] = field[1, i]
    field[n + 1, i] = field[n, i]
    field[i, 0] = field[i, 1]
    field[i, n + 1] = field[i, n] 

  field[0, 0] =     0.5 * (field[1, 0] + field[0, 1])
  field[0, n + 1] = 0.5 * (field[1, n + 1] + field[0, n])
  field[n + 1, 0] = 0.5 * (field[n, 0] + field[n + 1, 1])
  field[n + 1, n + 1] = 0.5 * (field[n, n + 1] + field[n + 1, n])


@ti.func
def bilinear_interpolate_nearest(x, y, field):
  i0 = int(x)
  j0 = int(y)
  i1 = i0 + 1
  j1 = j0 + 1

  right_weight = x - i0
  left_weight = 1 - right_weight
  top_weight = y - j0
  bottom_weight = 1 - top_weight
  
  interpolated_value = (
    left_weight * (bottom_weight * field[i0, j0] + top_weight * field[i0, j1]) + \
    right_weight * (bottom_weight * field[i1, j0] + top_weight * field[i1, j1])
  )

  return interpolated_value


@ti.kernel
def nullify_boundary_flow(n: int, h_velocity: ti.template(), v_velocity: ti.template()):

  for i in ti.ndrange((1, n + 1)):
    # reflex velicity at boundaries
    # top
    h_velocity[i, n + 1] = h_velocity[i, n]
    v_velocity[i, n + 1] = -v_velocity[i, n]

    # bottom
    h_velocity[i, 0] = h_velocity[i, 1]
    v_velocity[i, 0] = -v_velocity[i, 1]

    # left
    h_velocity[0, i] = -h_velocity[1, i]
    v_velocity[0, i] = v_velocity[1, i]

    # right
    h_velocity[n + 1, i] = -h_velocity[n, i]
    v_velocity[n + 1, i] = v_velocity[n, i]

  # adjust corners
  x0 = 0
  x1 = n + 1
  y0 = 0
  y1 = n + 1
  h_velocity[x0,y0] = 0.5 * (h_velocity[1, 0]     + h_velocity[0, 1])
  h_velocity[x0,y1] = 0.5 * (h_velocity[1, n + 1] + h_velocity[0, n])
  h_velocity[x1,y0] = 0.5 * (h_velocity[n, 0]     + h_velocity[n + 1, 1])
  h_velocity[x1,y1] = 0.5 * (h_velocity[n, n + 1] + h_velocity[n + 1, n])

  v_velocity[x0,y0] = 0.5 * (v_velocity[1, 0]     + v_velocity[0, 1])
  v_velocity[x0,y1] = 0.5 * (v_velocity[1, n + 1] + v_velocity[0, n])
  v_velocity[x1,y0] = 0.5 * (v_velocity[n, 0]     + v_velocity[n + 1, 1])
  v_velocity[x1,y1] = 0.5 * (v_velocity[n, n + 1] + v_velocity[n + 1, n])
