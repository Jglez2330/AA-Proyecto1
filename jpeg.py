# Se importan las librerías
import numpy as np  # Para las operaciones matemática
import cv2 as cv    # Para la exportación de imagenes
import scipy
# Para cargar una imagen
from IPython.display import Image
from google.colab import files
from google.colab.patches import cv2_imshow

# Método que carga la imagen de manera local o Google Drive
def upload_image():
  uploaded = files.upload()

  # Carga la imagen y la muestra
  for filename in uploaded.keys():
      print('\n Imagen original cargada:', filename)
      display(Image(filename=filename))

  # Ruta a la imagen cargada (asegúrate de que coincida con el nombre de archivo correcto)
  image_path = list(uploaded.keys())[0]

  # Carga la imagen usando cv2
  img = cv.imread(image_path, 0)

  # Verifica si la carga de la imagen fue exitosa
  if img is not None:
      print('Imagen cargada con éxito.')
      return img
      # Puedes realizar operaciones en la imagen aquí
  else:
      print('No se pudo cargar la imagen.')

# Matrices a utilizar
Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 59, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]
)

Q10 = np.array(
  [
    [80, 55, 50, 80, 120, 200, 255, 255],
    [60, 60, 70, 95, 130, 255, 255, 255],
    [70, 65, 80, 120, 200, 255, 255, 255],
    [70, 85, 110, 145, 255, 255, 255, 255],
    [90, 110, 185, 255, 255, 255, 255, 255],
    [120, 175, 255, 255, 255, 255, 255, 255],
    [245, 255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]
  ]
)

FAEQ10 = np.array(
  [
    [32, 32, 59, 160, 117, 255, 255, 255],
    [32, 32, 112, 178, 139, 255, 255, 255],
    [32, 32, 147, 186, 255, 255, 255, 255],
    [245, 245, 32, 225, 255, 255, 255, 255],
    [32, 109, 124, 255, 255, 255, 255, 255],
    [245, 115, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]
  ]
)

Q80 = np.array(
  [
    [6, 4, 4, 6, 10, 16, 20, 24],
    [5, 5, 6, 8, 10, 23, 24, 22],
    [6, 5, 6, 10, 16, 23, 28, 22],
    [6, 7, 9, 12, 20, 35, 32, 25],
    [7, 9, 15, 22, 27, 44, 41, 31],
    [10, 14, 22, 26, 32, 42, 45, 37],
    [20, 26, 31, 35, 41, 48, 48, 40],
    [29, 37, 38, 39, 45, 40, 41, 40]
  ]
)

Haddamark = np.array(
    [
        [16, 24, 16, 17, 16, 21, 16, 18],
        [24, 115, 36, 47, 25, 88, 29, 65],
        [16, 36, 21, 24, 16, 31, 18, 27],
        [17, 47, 24, 30, 17, 41, 20, 35],
        [16, 25, 16, 17, 16, 22, 16, 19],
        [21, 88, 31, 41, 22, 70, 25, 54],
        [16, 29, 18, 20, 16, 25, 17, 22],
        [18, 65, 27, 35, 19, 54, 22, 44]
    ]
)

Ultrasound = np.array(
    [
        [8, 16, 24, 40, 81, 97, 72, 145],
        [8, 16, 24, 40, 89, 162, 194, 283],
        [16, 16, 24, 40, 89, 170, 194, 283],
        [16, 24, 24, 40, 89, 170, 194, 291],
        [24, 24, 32, 48, 89, 162, 194, 275],
        [24, 32, 32, 48, 105, 178, 202, 299],
        [48, 48, 48, 72, 121, 194, 210, 291],
        [81, 81, 81, 89, 145, 202, 210, 291]
    ]
)

psycho_visual_threshold = np.array(
    [
        [16, 14, 13, 15, 19, 28, 37, 55],
        [14, 13, 15, 19, 28, 37, 55, 64],
        [13, 15, 19, 28, 37, 55, 64, 83],
        [15, 19, 28, 37, 55, 64, 83, 103],
        [19, 28, 37, 55, 64, 83, 103, 117],
        [28, 37, 55, 64, 83, 103, 117, 117],
        [37, 55, 64, 83, 103, 117, 117, 111],
        [55, 64, 83, 103, 117, 117, 111, 90]
    ]
)

matrix = {
    "Q50": Q50,
    "Q10": Q10,
    "FAEQ10": FAEQ10,
    "Q80": Q80,
    "Haddamark": Haddamark,
    "Ultrasound": Ultrasound,
    "psycho_visual_threshold": psycho_visual_threshold
}

def quantization(n):
  # This function gives the quantization, 8x8 matrix for a given n
  # n -> Level of quantization, it is always 50 for this experiment

  return matrix[n]

def zigzag(matrix):
  rows, cols = matrix.shape
  result = []

  for i in range(rows + cols - 1):
    if i % 2 == 0:  # Move upwards
      if i < rows:
        row, col = i, 0
      else:
        row, col = rows - 1, i - rows + 1

      while row >= 0 and col < cols:
        result.append(matrix[row, col])
        row -= 1
        col += 1

    else:  # Move downwards
      if i < cols:
        row, col = 0, i
      else:
        row, col = i - cols + 1, cols - 1

      while row < rows and col >= 0:
        result.append(matrix[row, col])
        row += 1
        col -= 1

  return np.array(result)

def inverse_zigzag(vector, rows, cols):
    matrix = np.zeros((rows, cols), dtype=vector.dtype)
    index = 0

    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Move upwards
            if i < rows:
                row, col = i, 0
            else:
                row, col = rows - 1, i - rows + 1
            while row >= 0 and col < cols:
                matrix[row, col] = vector[index]
                index += 1
                row -= 1
                col += 1
        else:  # Move downwards
            if i < cols:
                row, col = 0, i
            else:
                row, col = i - cols + 1, cols - 1
            while row < rows and col >= 0:
                matrix[row, col] = vector[index]
                index += 1
                row += 1
                col -= 1

    return matrix

def jpeg_compression(image, n_l):
  # This function compresses an image into a cell using the JPEG method
  # image -> matrix representing the image
  # n_l -> level of compression of the image
  # compressed_image -> cell representing the 8x8 blocks

  m, n = image.shape
  m = m // 8
  n = n // 8
  compressed_image = np.empty((m, n), dtype=object)

  for i in range(m):
    i_end = 8 * (i + 1)
    i_start = i_end - 8

    for j in range(n):
      j_end = 8 * (j  + 1)
      j_start = j_end - 8

      kernel = image[i_start:i_end, j_start:j_end]
      reduced_image = kernel - 128
      dct_image = scipy.fft.dctn(reduced_image, type=2, norm='ortho') # Discrete Cosine Transform
      Q = n_l                                                         # Quantization table
      quantized_image = np.round(dct_image / Q).astype(int)           # Quantization
      vect = zigzag(quantized_image)                                  # ZigZag Path

      compressed_image[i, j] = vect

  return compressed_image

def jpeg_decompression(compressed_image, n_l):
  # This function decompresses an image into a cell using the JPEG method
  # compressed_image -> cell representing the 8x8 blocks of the compressed image
  # n_l -> level of compression of the image
  # decompressed_image -> matrix representing the image

  m, n = compressed_image.shape
  decompressed_image = np.zeros((m * 8, n * 8), dtype=int)

  for i in range(m):
    i_end = 8 * (i + 1)
    i_start = i_end - 8

    for j in range(n):
      j_end = 8 * (j + 1)
      j_start = j_end - 8

      kernel = compressed_image[i, j]
      i_zigzag = inverse_zigzag(kernel, 8, 8)
      Q = n_l
      kernel_image = Q * i_zigzag

      rounded_kernel = np.round(scipy.fft.idctn(kernel_image, type=2, norm='ortho')) #Inverse discrete cosine transform
      rounded_kernel = rounded_kernel + 128
      decompressed_image[i_start:i_end, j_start:j_end] = rounded_kernel

  return decompressed_image

# Función para determinar la métrica de error
# Se utilizo Error Cuadrático Medio (MSE - Mean Squared Error)
# El MSE calcula el promedio de los cuadrados de las diferencias entre los valores de píxeles de la imagen original
# y la imagen comprimida y descomprimida. Cuanto menor sea el MSE, mejor será la calidad de la imagen.
def calculate_mse(image1, image2):
    # Redimensionar una de las imágenes para que coincida con el tamaño de la otra
    if image1.shape != image2.shape:
        image2 = cv.resize(image2, (image1.shape[1], image1.shape[0]))

    return np.mean((image1 - image2) ** 2)

# Función para ejecutar la imagen por todas las matrices
def image_x_matrix(imagen):
  # Iterar a través de cada matriz de cuantización en el diccionario 'matrix'
  for key, quantization_matrix in matrix.items():
    # Realizar la compresión
    compressed_img = jpeg_compression(imagen, quantization_matrix)
    
    # Realizar la descompresión
    decompressed_img = jpeg_decompression(compressed_img, quantization_matrix)
    
    # Asegurarse de que las imágenes estén en el rango adecuado (0-255)
    imagen = np.uint8(imagen)
    decompressed_img = np.uint8(decompressed_img)
    
    # Calcular el MSE entre la imagen original y la imagen descomprimida
    mse = calculate_mse(imagen, decompressed_img)
    
    print(f"\nMatriz {key}")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f} %")
    cv2_imshow(decompressed_img)
