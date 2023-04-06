import taichi as ti

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

  @ti.kernel
  def swap(self):
    for i, j in self.current:
      temp = self.previous[i, j]
      self.previous[i, j] = self.current[i, j]
      self.current[i, j] = temp