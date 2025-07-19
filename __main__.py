import os
from PIL import Image

from .src import MODEL, vis_segmentation

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def run_visualization(SAMPLE_IMAGE):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(os.path.join(ROOT_DIR, "images/", SAMPLE_IMAGE))
    seg_map = MODEL.run(original_im)
    output_path = f"resultados/segmentacao_{0:03}.png" 
    vis_segmentation(original_im, seg_map, output_path, True)

def main():
    # SAMPLE_IMAGE ="fig1.png" 
    # SAMPLE_IMAGE ="fig2.png" 
    # SAMPLE_IMAGE ="fig3.png" 
    SAMPLE_IMAGE = "fig4.png" 
    
    run_visualization(SAMPLE_IMAGE)


if __name__ == "__main__":
    main()
