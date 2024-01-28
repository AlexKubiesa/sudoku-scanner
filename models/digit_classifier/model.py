from pathlib import Path
import tensorflow as tf


class DigitClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(Path(__file__).parent / "keras")

    def __call__(self, inputs):
        """Classifies a Sudoku cell image.

        Arguments:
            inputs (tf.Tensor): A tensor of shape (H, W, C) where H and W are the height
                and width of the image respectively, and C is the number of channels.
                The channels are expected to be in the order RGB. The tensor values are
                expected to be floating-point numbers in the range [0, 1].

        Returns:
            tf.Tensor: A tensor of shape () containing the integer class prediction. See
                the `end_to_end` dataset readme for descriptions of the classes.
        """
        assert inputs.ndim == 3, "Expected a 3D tensor of shape (H, W, C)"
        inputs = tf.expand_dims(inputs, axis=0)
        outputs = self.model(inputs)
        outputs = tf.squeeze(outputs, axis=0)
        outputs = tf.argmax(outputs)
        return outputs
