import imageio.v3 as imageio
from numpy import ndarray

def load_image(path: str) -> ndarray:
  image = imageio.imread(path)
  return image

def convert_to_greyscale(image: ndarray) -> ndarray:
  return image.mean(axis=2)

if __name__ == "__main__":
  image = load_image("src/assets/test.jpg")
  greyscale_image = convert_to_greyscale(image)
  print(greyscale_image.shape)