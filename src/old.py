import taichi as ti
import taichi.math as tim

ti.init(arch=ti.gpu)

n = 1000
canvas_size = (n, n)
pixels = ti.field(dtype=float, shape=canvas_size)

@ti.func
def get_adjacent(i, j, source):
    left = source[i - 1, j]
    right = source[i + 1, j]
    up = source[i, j - 1]
    down = source[i, j + 1]

    return (left, right, up, down)


@ti.func
def contain(n, density):
    for i in ti.ndrange((1, n + 1)):
        # contain
        density[0, i] = density[1, i]
        density[n + 1, i] = density[n, i]
        density[i, 0] = density[i, 1]
        density[i, n + 1] = density[i, n]

    # adjust corners
    density[0, 0] = 0.5 * (density[1, 0] + density[0, 1])
    density[0, n + 1] = 0.5 * (density[1, n + 1] + density[0, n])
    density[n + 1, 0] = 0.5 * (density[n, 0] + density[n + 1, 1])
    density[n + 1, n + 1] = 0.5 * (density[n, n + 1] + density[n + 1, n])


@ti.func
def nullify_boundary_flow(n, hori_vel, vert_vel):
    for i in ti.ndrange((1, n + 1)):
        # top
        hori_vel[i, n + 1] = hori_vel[i, n]
        vert_vel[i, n + 1] = -vert_vel[i, n]

        # bottom
        hori_vel[i, 0] = hori_vel[i, 1]
        vert_vel[i, 0] = -vert_vel[i, 1]

        # left
        hori_vel[0, i] = -hori_vel[1, i]
        vert_vel[0, i] = vert_vel[1, i]

        # right
        hori_vel[n + 1, i] = -hori_vel[n, i]
        vert_vel[n + 1, i] = vert_vel[n, i]

    # adjust corners
    hori_vel[0, 0] = 0.5 * (hori_vel[1, 0] + hori_vel[0, 1])
    hori_vel[0, n + 1] = 0.5 * (hori_vel[1, n + 1] + hori_vel[0, n])
    hori_vel[n + 1, 0] = 0.5 * (hori_vel[n, 0] + hori_vel[n + 1, 1])
    hori_vel[n + 1, n + 1] = 0.5 * (hori_vel[n, n + 1] + hori_vel[n + 1, n])

    vert_vel[0, 0] = 0.5 * (vert_vel[1, 0] + vert_vel[0, 1])
    vert_vel[0, n + 1] = 0.5 * (vert_vel[1, n + 1] + vert_vel[0, n])
    vert_vel[n + 1, 0] = 0.5 * (vert_vel[n, 0] + vert_vel[n + 1, 1])
    vert_vel[n + 1, n + 1] = 0.5 * (vert_vel[n, n + 1] + vert_vel[n + 1, n])
        
        


@ti.data_oriented
class TemporalValueField:
    def __init__(self, shape, dtype) -> None:
        self.current = ti.field(dtype=dtype, shape=shape)
        self.previous = ti.field(dtype=dtype, shape=shape)

    @ti.func
    def set(self, i, j, value):
        self.current[i, j] = value

    @ti.func
    def get_prev(self, i, j):
        return self.previous[i, j]

    @ti.func
    def get(self, i, j):
        return self.current[i, j]

    @ti.func
    def swap(self):
        for i, j in self.current:
            temp = self.previous[i, j]
            self.previous[i, j] = self.current[i, j]
            self.current[i, j] = temp

    @ti.func
    def get_adjacent(self, i, j):
        return get_adjacent(i, j, self.current)
    
    @ti.func
    def get_adjacent_prev(self, i, j):
        return get_adjacent(i, j, self.previous)
    

@ti.func
def diffuse(grid_size, value_field, viscosity, dt):
    diffusion_rate = dt * viscosity * grid_size * grid_size

    ti.block_local(value_field.previous)
    for i, j in ti.ndrange((1, grid_size - 1), (1, grid_size - 1)):
        for _ in range(20):
            left, right, up, down = value_field.get_adjacent_prev(i, j)

            surrounding_density = left + right + up + down
            absorb_diffusion = diffusion_rate * surrounding_density

            prev_density = value_field.get_prev(i, j)
            numerator = prev_density + absorb_diffusion
            denominator = 1 + 4 * diffusion_rate

            value_field.set(i, j, numerator / denominator)


@ti.func
def interpolate_values(x, y, source):
    # bottom left corner of cell that contains the particle
    nearest_cell_i = tim.round(x, dtype=int)
    nearest_cell_j = tim.round(y, dtype=int)

    # distance from bottom left corner of cell to particle
    s = x - nearest_cell_i
    t = y - nearest_cell_j

    return \
        (1 - s) * (1 - t) * source[nearest_cell_i,      nearest_cell_j] + \
        (  s  ) * (  t  ) * source[nearest_cell_i + 1,  nearest_cell_j + 1] + \
        (1 - s) * (  t  ) * source[nearest_cell_i,      nearest_cell_j + 1] + \
        (  s  ) * (1 - t) * source[nearest_cell_i + 1,  nearest_cell_j]


@ti.func
def advect(grid_size, target_field, hori_vel, vert_vel, dt):
    dt_scaled = dt * grid_size
    for i, j in ti.ndrange((1, grid_size + 1), (1, grid_size + 1)):
        particle_start_x = i - dt_scaled * hori_vel[i, j]
        particle_start_y = j - dt_scaled * vert_vel[i, j]

        # Clamp to within bounds
        particle_start_x = tim.max(0.5, tim.max(grid_size + 0.5, particle_start_x))
        particle_start_y = tim.max(0.5, tim.max(grid_size + 0.5, particle_start_y))

        target_field.set(i, j, interpolate_values(particle_start_x, particle_start_y, target_field.previous))


@ti.func
def add_source(target_field, source_field, dt):
    for i, j in target_field:
        target_field[i, j] += dt * source_field[i, j]


@ti.func
def project(grid_size, hori_vel_field, vert_vel_field, pressure, divergence):
    n_factor = 1 / grid_size
    for i,j in ti.ndrange((1, grid_size + 1), (1, grid_size + 1)):
        left, right, _, _ = get_adjacent(i, j, hori_vel_field)
        _, _, up, down = get_adjacent(i, j, vert_vel_field)

        divergence[i, j] = -0.5 * n_factor * (right - left + up - down)
        pressure[i, j] = 0

    contain(grid_size, divergence)
    contain(grid_size, pressure)

    for i, j in ti.ndrange((1, grid_size + 1), (1, grid_size + 1)):
        for _ in range(20):
            left, right, up, down = get_adjacent(i, j, pressure)
            pressure[i, j] = (divergence[i, j] + left + right + up + down) / 4

    contain(grid_size, pressure)

    for i, j in ti.ndrange((1, grid_size + 1), (1, grid_size + 1)):
        left, right, up, down = get_adjacent(i, j, pressure)

        hori_vel_field[i, j] -= 0.5 * (right - left) / n_factor
        vert_vel_field[i, j] -= 0.5 * (up - down) / n_factor


@ti.data_oriented
class FluidField:
    def __init__(self, n):
        self.n = n

        self.boundry_layer = 2
        field_size = n + self.boundry_layer

        self.density = TemporalValueField((field_size, field_size), float)
        self.hori_vel = TemporalValueField((field_size, field_size), float)
        self.vert_vel = TemporalValueField((field_size, field_size), float)

        self.pressure = ti.field(dtype=float, shape=(field_size, field_size))
        self.divergence = ti.field(dtype=float, shape=(field_size, field_size))
    
    def step(self, dt: float):
        self.step_velocity(dt)
        self.step_density(dt)

    @ti.kernel
    def step_velocity(self, dt: float):
        self.hori_vel.swap()
        diffuse(self.n, self.hori_vel, 100, dt)
        
        self.vert_vel.swap()
        diffuse(self.n, self.vert_vel, 100, dt)

        project(self.n, self.hori_vel.current, self.vert_vel.current, self.pressure, self.divergence)
        nullify_boundary_flow(self.n, self.hori_vel.current, self.vert_vel.current)

        self.hori_vel.swap()
        self.vert_vel.swap()

        advect(self.n, self.hori_vel, self.hori_vel.previous, self.vert_vel.previous, dt)
        advect(self.n, self.vert_vel, self.hori_vel.previous, self.vert_vel.previous, dt)
        nullify_boundary_flow(self.n, self.hori_vel.current, self.vert_vel.current)

        project(self.n, self.hori_vel.current, self.vert_vel.current, self.pressure, self.divergence)
        nullify_boundary_flow(self.n, self.hori_vel.current, self.vert_vel.current)

    @ti.kernel
    def step_density(self, dt: float):
        # add_source(self.density.current, self.density.previous, dt)
        self.density.swap()
        diffuse(self.n, self.density, 100, dt)
        contain(self.n, self.density.current)

        self.density.swap()
        advect(self.n, self.density, self.hori_vel.current, self.vert_vel.current, dt)
        contain(self.n, self.density.current)




fluid = FluidField(n)
gui = ti.GUI("2D Fluid Simulation", canvas_size)


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = tim.min(fluid.density.current[i, j], 10)


while not gui.get_event(ti.GUI.ESCAPE):
    if gui.is_pressed(ti.GUI.LMB):
        mouse_x, mouse_y = gui.get_cursor_pos()
        fluid.density.current[int(mouse_x * n), int(mouse_y * n)] = 10

    fluid.step(0.1)
    render()
    gui.set_image(pixels)
    gui.show()