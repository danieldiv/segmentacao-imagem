import os
import time
from PIL import Image

from .src import load_model, vis_segmentation

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def run_visualization(SAMPLE_IMAGE):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(os.path.join(ROOT_DIR, "images/", SAMPLE_IMAGE))

    # MODEL = load_model("mobilenet")
    MODEL = load_model("xception")

    inicio = time.time()
    seg_map = MODEL.run(original_im)
    fim = time.time()
    print(f"Tempo de processamento: {fim - inicio:.4f} segundos")

    output_path = f"resultados/segmentacao_{0:03}.png"
    vis_segmentation(original_im, seg_map, output_path, True)


def main():
    # SAMPLE_IMAGE ="fig1.png"
    # SAMPLE_IMAGE ="fig2.png"
    SAMPLE_IMAGE = "fig3.png"
    # SAMPLE_IMAGE = "fig4.png"

    run_visualization(SAMPLE_IMAGE)


if __name__ == "__main__":
    main()

# import os
# import time
# import gradio as gr
# from functools import partial

# from .src import segment_image


# def stop_server():
#     print("Encerrando o servidor...")
#     time.sleep(1)
#     os._exit(0)


# def main():
#     # ARQUITETURA = "mobilenet"
#     ARQUITETURA = "xception"

#     segment_with_arch = partial(segment_image, arquitetura=ARQUITETURA)

#     with gr.Blocks() as demo:
#         gr.Markdown("## Segmentação com MobileNet")
#         gr.Markdown("Envie uma imagem urbana para segmentação:")

#         with gr.Row():
#             with gr.Column():
#                 image_input = gr.Image(type="pil", label="Imagem de entrada")
#                 btn_segment = gr.Button("Segmentar")
#                 btn_close = gr.Button("Fechar Servidor")

#             image_output = gr.Image(type="pil", label="Segmentação")

#         btn_segment.click(
#             fn=segment_with_arch, inputs=image_input, outputs=image_output
#         )
#         btn_close.click(fn=stop_server, inputs=[], outputs=[])

#     demo.launch()


# if __name__ == "__main__":
#     main()
