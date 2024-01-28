from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


def get_data(split):
    """
    Generate and yield image data and corresponding labels for a given split.

    Args:
        split (str): The split of the dataset to get the data from. Possible values
            are "train", "val", and "test".

    Yields:
        tuple: A tuple containing the image data and corresponding labels.
    """
    path = Path("datasets") / "end_to_end" / split

    for image_path in (path / "images").iterdir():
        image = tf.keras.preprocessing.image.load_img(image_path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0

        labels_path = path / "labels" / (image_path.stem + ".txt")
        labels = np.loadtxt(labels_path, dtype=np.int32)

        yield image, labels
