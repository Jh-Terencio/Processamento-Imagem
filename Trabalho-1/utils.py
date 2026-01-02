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

def manual_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica convolução 2D manual (usando laços) entre uma imagem em tons de cinza
    e um kernel (máscara) arbitrário.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem em tons de cinza, com shape (altura, largura).
    kernel : np.ndarray
        Máscara 2D da convolução. Deve ter dimensões ímpares, por exemplo 3x3, 5x5, etc.

    Retorno
    -------
    output : np.ndarray
        Imagem resultante da convolução, mesma dimensão da imagem de entrada.
    """
     
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    k_h, k_w = kernel.shape      # altura e largura do kernel
    img_h, img_w = image.shape   # altura e largura da imagem

    # Metade do tamanho do kernel, usado para o padding
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Faz padding com zeros ao redor da imagem
    # Assim conseguimos aplicar a máscara também nas bordas
    padded = np.pad(
        image,
        pad_width=((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    # Cria a imagem de saída
    output = np.zeros_like(image, dtype=np.float32)

    # Percorre cada pixel da imagem original
    for i in range(img_h):
        for j in range(img_w):
            # Recorta da imagem "padded" a vizinhança coberta pelo kernel
            region = padded[i:i + k_h, j:j + k_w]

            # Produto elemento a elemento + soma -> operação de convolução
            value = np.sum(region * kernel)

            # Atribui o resultado ao pixel de saída
            output[i, j] = value

    # Depois da convolução, garantimos que os valores fiquem no intervalo [0, 255]
    output = np.clip(output, 0, 255)

    # Converte de volta para uint8 (imagem normal de 8 bits)
    output = output.astype(np.uint8)

    return output


def create_random_kernel(mask_size: int, low: int = -2, high: int = 3) -> np.ndarray:
    """
    Cria um kernel (máscara) com valores inteiros aleatórios.

    Parâmetros
    ----------
    mask_size : int
        Tamanho da máscara (deve ser um número ímpar: 3, 5, 7, ...).
    low : int, opcional
        Valor inteiro mínimo (inclusivo) para os elementos do kernel.
    high : int, opcional
        Valor inteiro máximo (exclusivo) para os elementos do kernel.
        Ex.: low=-2, high=3 -> valores em {-2, -1, 0, 1, 2}.
    normalize : bool, opcional
        Se True, normaliza o kernel para que a soma dos elementos seja 1.
        Isso ajuda a evitar que a imagem fique muito clara/escura.

    Retorno
    -------
    kernel : np.ndarray
        Kernel 2D (mask_size x mask_size) com valores inteiros (ou float, se normalizado).
    """
    if mask_size <= 0:
        raise ValueError("O tamanho da máscara deve ser positivo.")
    if mask_size % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar (ex.: 3, 5, 7).")

    # Cria matriz de inteiros aleatórios entre [low, high)
    kernel = np.random.randint(low=low, high=high,
                               size=(mask_size, mask_size),
                               dtype=np.int32)

    return kernel



def automatic_convolution(gray_image: np.ndarray, mask_size: int, random_seed: bool = False) -> np.ndarray:
    """
    1. Cria automaticamente um kernel de média do tamanho especificado.
    2. Aplica a convolução manual na imagem em tons de cinza.

    Parâmetros
    ----------
    gray_image : np.ndarray
        Imagem em tons de cinza (1 canal).
    mask_size : int
        Tamanho da máscara de convolução (ímpar).

    Retorno
    -------
    filtered_image : np.ndarray
        Imagem resultante após a aplicação do filtro de média.
    """
    if random_seed:
        np.random.seed(42)
        
    kernel = create_random_kernel(mask_size, low=0, high=2)
    
    print("Kernel usado na convolução:")
    print(kernel)
    
    filtered_image = manual_convolution(gray_image, kernel)
    return filtered_image

import numpy as np

def apply_laplacian_filter(gray_image: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro Laplaciano (kernel do professor) em uma imagem em tons de cinza.
    """
    image = gray_image.astype(np.float32)

    # Kernel laplaciano conforme mostrado em aula
    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    k_h, k_w = kernel.shape
    img_h, img_w = image.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(
        image,
        pad_width=((pad_h, pad_h), (pad_w, pad_w)),
        mode="edge"
    )

    response = np.zeros_like(image, dtype=np.float32)

    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j:j + k_w]
            response[i, j] = np.sum(region * kernel)

    # valor absoluto para visualizar bordas (positivo/negativo somem)
    response = np.abs(response)

    max_val = response.max()
    if max_val > 0:
        response = response / max_val * 255.0

    edges = response.astype(np.uint8)
    return edges


def show_images_side_by_side(original: np.ndarray, filtered: np.ndarray, mask_size: int) -> None:
    """
    Mostra a imagem original e a imagem filtrada lado a lado usando Matplotlib.

    Parâmetros
    ----------
    original : np.ndarray
        Imagem original em tons de cinza.
    filtered : np.ndarray
        Imagem após a convolução.
    mask_size : int
        Tamanho da máscara utilizada (apenas para título do gráfico).
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Imagem original (gray)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap="gray")
    plt.title(f"Imagem filtrada\nMáscara {mask_size}x{mask_size}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def manual_erosion(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erosão morfológica manual
    - Para imagem binária (0/255): pixel sai branco (255) apenas se a região coberta pelo SE for toda branca.
    - Para imagem em tons de cinza: faz erosão em cinza (mínimo sob o SE).

    Parâmetros
    ----------
    image : np.ndarray
        Imagem 2D (binária 0/255 ou tons de cinza).
    se : np.ndarray
        Elemento estruturante 2D (0/1), ex.: 3x3.
    iterations : int
        Número de iterações.

    Retorno
    -------
    np.ndarray
        Imagem erodida (uint8).
    """
    if image.ndim != 2:
        raise ValueError("manual_erosion espera uma imagem 2D (1 canal).")

    out = image.astype(np.uint8)
    se = (se > 0).astype(np.uint8)

    k_h, k_w = se.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    se_mask = (se == 1)

    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode="constant", constant_values=0)

        eroded = np.zeros_like(out, dtype=np.uint8)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                region = padded[i:i + k_h, j:j + k_w]
                # erosão = mínimo apenas onde o SE "vale 1"
                eroded[i, j] = np.min(region[se_mask])

        out = eroded

    return out

def manual_dilation(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Dilatação morfológica manual

    Observação importante (binário 0/255)
    ------------------------------------
    Esta implementação assume o caso mais comum em morfologia binária:
    - OBJETO (foreground) = 255 (branco)
    - FUNDO (background)  = 0   (preto)

    Se o seu objeto estiver preto em fundo branco, inverta antes:
        binary = 255 - binary
    """
    if image.ndim != 2:
        raise ValueError("manual_dilation espera uma imagem 2D (1 canal).")

    out = image.astype(np.uint8)
    se = (se > 0).astype(np.uint8)

    k_h, k_w = se.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    se_mask = (se == 1)

    background_value = 0  # fundo preto (0)

    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode="constant", constant_values=background_value)

        dilated = np.zeros_like(out, dtype=np.uint8)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                region = padded[i:i + k_h, j:j + k_w]
                dilated[i, j] = np.max(region[se_mask])

        out = dilated

    return out