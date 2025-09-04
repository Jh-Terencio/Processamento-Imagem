import numpy as np
import cv2 as cv

def histograma(matriz, posicao=None):
    if posicao == None:
        array = 256*[0]
        for i in range (matriz.shape[0]):
            for j in range (matriz.shape[1]):
                valor = matriz[i,j]
                array[valor] += 1
    else:
        array = 256*[0]
        for i in range (matriz.shape[0]):
            for j in range (matriz.shape[1]):
                valor = matriz[i,j][posicao]
                array[valor] += 1
                
    return array
  
def eda_image(image: np.ndarray):
  # ---- Resolução da imagem -------
  # Em OpenCV, a ordem do shape é (altura, largura, canais)
  # Altura  -> número de linhas (pixels na vertical)
  # Largura -> número de colunas (pixels na horizontal)

  print('Largura em pixels: ', end='')   # Número de linhas (altura)
  print(image.shape[0])

  print('Altura em pixels: ', end='')    # Número de colunas (largura)
  print(image.shape[1])

  # Quantidade total de pixels (sem contar os canais)
  print('Quantidade de pixels: ', end='')
  print(image.shape[0] * image.shape[1])

  # Quantidade de elementos na matriz (altura * largura * canais)
  # Similar ao produto: image.shape[0] * image.shape[1] * image.shape[2]
  print("Quantidade de elementos presentes na matriz: ", end='')
  print(image.size)
  # ---------------------------------

  # Quantidade de canais da imagem
  # Imagem colorida RGB -> 3 canais
  # Imagem em escala de cinza -> 1 canal
  print('Quantidade de canais: ', end='')
  print(image.shape[2])
  
  cv.imshow("Janela da image", image)
  cv.waitKey(0)
  
def generate_gray_channel(image: np.ndarray) -> np.ndarray:
  grayChannel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

  for i in range(image.shape[0]):      # percorre as linhas (altura)
      for j in range(image.shape[1]):  # percorre as colunas (largura)
          # Média simples dos três canais (B+G+R)/3
          grayChannel[i, j] = (image[i, j].sum() // 3)
  
  return grayChannel

def remove_threshold(matriz: np.ndarray, threshold, mode: str = 'above') -> np.ndarray:
    """
    Se threshold for int:
        - mode='above': mantém apenas valores > threshold
        - mode='below': mantém apenas valores < threshold
    Se threshold for list:
        - threshold = [(min1, max1), (min2, max2), ...]
        - mantém apenas valores dentro dos intervalos
    Os que não atendem viram 255.
    """
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            value = matriz[i, j]
            if isinstance(threshold, int):
                if mode == 'above':
                    if value <= threshold:
                        matriz[i, j] = 255
                elif mode == 'below':
                    if value >= threshold:
                        matriz[i, j] = 255
                else:
                    raise ValueError("mode deve ser 'above' ou 'below'")
            elif isinstance(threshold, list):
                keep = any(min_ <= value <= max_ for (min_, max_) in threshold)
                if not keep:
                    matriz[i, j] = 255
            else:
                raise TypeError("threshold deve ser int ou list de tuplas")
    return matriz