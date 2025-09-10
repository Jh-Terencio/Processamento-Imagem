import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv  # assumindo que você já importou cv2

def manipulate_contrast_luminosity(matriz: np.ndarray, contrast=0, luminosity=0, mode: str = 'normal') -> np.ndarray:
    """
    Função para manipular contraste e luminosidade de uma imagem em escala de cinza.
    Pode aplicar transformação parabólica, inversão ou ajuste linear.
    Parâmetros:
    - matriz: np.ndarray - imagem em escala de cinza.
    - contrast: float - fator de contraste (usado no modo 'normal').
    - luminosity: float - valor de luminosidade (usado no modo 'normal').
    - mode: str, opcional (default='normal')
        Modo de transformação:
        - 'parabolica': Aplica uma transformação parabólica aos pixels. Multiplicamos por 255 para garantir que o resultado esteja na faixa de 0 a 255, pois a operação ((r/255)^2) gera valores entre 0 e 1.
        - 'invertido': Inverte os valores dos pixels (negativo da imagem).
        - 'normal': Aplica ajuste linear de contraste e luminosidade.
    Retorno:
    - np.ndarray
        Imagem transformada após a manipulação de contraste/luminosidade.
    Notas:
    - A função exibe o histograma da imagem resultante e a curva de transformação utilizada.
    - A multiplicação por 255 na transformação parabólica serve para normalizar o resultado para o intervalo de intensidade de pixels (0 a 255).
    """
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            r = int(matriz[i, j])
            if mode == 'parabolica':
                
                s = (((1/256)*r)**2) * 255
            elif mode == 'invertido':
                s = 255 - r
            elif mode == 'normal':
                if contrast == 0:
                    s = r + luminosity
                else:
                    s = (contrast * r) + luminosity

            if s > 255:
                matriz[i, j] = 255
            else:
                matriz[i, j] = s

    pixel = np.arange(256)

    # Preparar figura com 2 gráficos lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1 - Histograma
    axs[0].bar(pixel, histograma(matriz), alpha=0.5)
    axs[0].set_title('Histograma da imagem')
    axs[0].set_xlabel('Valor do pixel')
    axs[0].set_ylabel('Frequência')

    # Gráfico 2 - Curva de transformação
    input_pixels = np.arange(256)
    
    if mode == 'parabolica':
        output_pixels = ((input_pixels / 255) ** 2) * 255
        titulo = 'Curva Paraboloide'
    elif mode == 'invertido':
        output_pixels = 255 - input_pixels
        titulo = 'Curva negativa'
    elif mode == 'normal':
        output_pixels = contrast * input_pixels + luminosity
        titulo = 'Curva de Transformação de Contraste e Luminosidade'

    axs[1].plot(input_pixels, output_pixels, color='blue', linewidth=2)
    axs[1].set_title(titulo)
    axs[1].set_xlabel('Valor Original do Pixel')
    axs[1].set_ylabel('Valor após Transformação')
    axs[1].grid(True)
    axs[1].set_xlim(0, 270)
    axs[1].set_ylim(0, 270)

    plt.tight_layout()
    plt.show()

    cv.imshow("Janela da imagem", matriz)
    cv.waitKey(0)

    return matriz
