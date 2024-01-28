from pathlib import Path
import pandas as pd
import tensorflow as tf


def get_data(split):
    path = Path("datasets") / "puzzle_localization" / split

    labels = pd.read_csv(path / "labels.csv")
    for _, row in labels.iterrows():
        image_path = path / "images" / row["Name"]
        image = tf.keras.preprocessing.image.load_img(image_path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        # No scaling down by 255.0 because the model is tuned to [0, 255] images.

        keypoints = tf.constant(
            row[
                [
                    "x_topleft",
                    "y_topleft",
                    "x_bottomleft",
                    "y_bottomleft",
                    "x_bottomright",
                    "y_bottomright",
                    "x_topright",
                    "y_topright",
                ]
            ].to_numpy(dtype=float)
        )

        yield image, keypoints
