import tensorflow as tf

print()

import os

import tarfile
import numpy as np
from PIL import Image
import cv2


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        if graph_def is None:
            raise RuntimeError("Cannot find inference graph in tar archive.")

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(
        self,
        image,
        INPUT_TENSOR_NAME="ImageTensor:0",
        OUTPUT_TENSOR_NAME="SemanticPredictions:0",
    ):
        """Runs inference on a single image."""
        width, height = image.size
        target_size = (2049, 1025)  # size of Cityscapes images

        # Compatível com Pillow novo e antigo
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        resized_image = image.convert("RGB").resize(target_size, resample)
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
        )
        seg_map = batch_seg_map[0]  # expected batch size = 1
        if len(seg_map.shape) == 2:
            seg_map = np.expand_dims(
                seg_map, -1
            )  # need an extra dimension for cv2.resize
        seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)
        return seg_map
