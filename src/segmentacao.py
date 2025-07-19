import matplotlib.pyplot as plt

# from IPython.display import clear_output
# import matplotlib.gridspec as gridspec

import os
import numpy as np


def create_label_colormap():
    colormap = np.array(
        [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )  # cores da minha segmentação
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def vis_segmentation(image, seg_map, save_path=None, show=False):
    """Salva imagem original + segmentada com pequeno espaço branco entre elas (layout manual)."""
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    # Tamanho da imagem em polegadas (100 dpi)
    width_in = image.width / 100
    height_in = image.height / 100
    separacao_frac = 0.005  # Pequeno espaço entre imagens (0.5% da altura)

    # Calcula altura das imagens com espaço entre elas
    total_height = 2 + separacao_frac  # 1 imagem + 1 imagem + espaço
    h_img = 1 / total_height
    h_sep = separacao_frac / total_height

    fig = plt.figure(figsize=(width_in, height_in * 2), dpi=100)

    # Imagem original (parte de cima)
    ax1 = fig.add_axes([0, h_img + h_sep, 1, h_img])  # [x, y, width, height]
    ax1.imshow(image)
    ax1.axis("off")

    # Segmentação sobreposta (parte de baixo)
    ax2 = fig.add_axes([0, 0, 1, h_img])
    ax2.imshow(image)
    ax2.imshow(seg_image, alpha=0.7)
    ax2.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Salvo em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# classes do meu modelo
LABEL_NAMES = np.asarray(
    [
        "estrada",  # road
        "calçada",  # sidewalk
        "prédio",  # building
        "muro",  # wall
        "cerca",  # fence
        "poste",  # pole
        "semáforo",  # traffic light
        "placa de trânsito",  # traffic sign
        "vegetação",  # vegetation
        "terreno",  # terrain
        "céu",  # sky
        "pessoa",  # person
        "motociclista",  # rider
        "carro",  # car
        "caminhão",  # truck
        "ônibus",  # bus
        "trem",  # train
        "motocicleta",  # motorcycle
        "bicicleta",  # bicycle
        "vazio",  # void
    ]
)


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
