import os
import time
import cv2

from PIL import Image
from .src import load_model, vis_segmentation, run_visualization_video

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def create_directories():
    """
    Cria os diretórios necessários para armazenar os resultados,
    modelos e imagens, caso ainda não existam.
    """
    resultados_path = os.path.join(ROOT_DIR, "resultados")
    models_path = os.path.join(ROOT_DIR, "models")
    images_path = os.path.join(ROOT_DIR, "images")
    videos_path = os.path.join(ROOT_DIR, "videos")

    os.makedirs(resultados_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(videos_path, exist_ok=True)


def run_visualization(SAMPLE_IMAGE, MODEL):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(os.path.join(ROOT_DIR, "images/", SAMPLE_IMAGE))

    inicio = time.time()
    seg_map = MODEL.run(original_im)
    fim = time.time()
    print(f"Tempo de processamento: {fim - inicio:.4f} segundos")

    output_path = os.path.join(ROOT_DIR, f"resultados/segmentacao_{0:03}.png")
    vis_segmentation(original_im, seg_map, output_path, True)


def main():
    MODEL = load_model("mobilenet")
    # MODEL = load_model("xception")

    # SAMPLE_IMAGE = "fig2.png"

    # create_directories()
    # run_visualization(SAMPLE_IMAGE, MODEL)
    # return

    SAMPLE_VIDEO = os.path.join(ROOT_DIR, "videos/video-rua.mp4")
    # SAMPLE_VIDEO = os.path.join(ROOT_DIR, "videos/video.mp4")

    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        exit()

    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    AMOSTRAS = 5
    print(TOTAL_FRAMES)

    # Calcula os índices dos frames desejados (uniformemente espaçados)
    frame_indices = [int(i * TOTAL_FRAMES / AMOSTRAS) for i in range(AMOSTRAS)]
    print(frame_indices)

    for i, frame_number in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Vai direto ao frame desejado
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break

        run_visualization_video(frame, i, MODEL)
        i = i + 1
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
