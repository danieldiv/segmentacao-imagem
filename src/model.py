from .deepLabModel import DeepLabModel

import os
from six.moves import urllib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("sempre roda")
USE_MOBILENET = True

if USE_MOBILENET:
    MODEL_NAME = "mobilenetv2_coco_cityscapes_trainfine"  # usa um modelo de dados urbanos, ruas, predios etc
    model_tar = "mobilenet.tar.gz"
else:
    MODEL_NAME = "xception65_cityscapes_trainfine"
    model_tar = "xception.tar.gz"

# MODEL_NAME = "xception65_cityscapes_trainfine"

_DOWNLOAD_URL_PREFIX = "http://download.tensorflow.org/models/"
_MODEL_URLS = {
    "mobilenetv2_coco_cityscapes_trainfine": "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz",
    "xception65_cityscapes_trainfine": "deeplabv3_cityscapes_train_2018_02_06.tar.gz",
}

download_path = os.path.join(f"{ROOT_DIR}/models", model_tar)

if not os.path.exists(download_path):
    # Baixa só se não existir
    print("downloading model, this might take a while...")
    urllib.request.urlretrieve(
        _DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path
    )
    print("download completed! loading DeepLab model...")
else:
    print("Model file already exists, skipping download.")

print("loading DeepLab model from:", download_path)
MODEL = DeepLabModel(download_path)
print("model loaded successfully!")
