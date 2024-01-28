import numpy as np
import cv2 as cv
import tensorflow as tf

from ..puzzle_localizer.model import HeuristicPuzzleLocalizer
from ..digit_classifier.model import DigitClassifier


PUZZLE_TARGET_SIZE = 576  # 64 * 9
PUZZLE_TARGET_KEYPOINTS = np.array(
    [
        [0, 0],
        [0, PUZZLE_TARGET_SIZE],
        [PUZZLE_TARGET_SIZE, PUZZLE_TARGET_SIZE],
        [PUZZLE_TARGET_SIZE, 0],
    ],
    dtype=np.float32,
)

CELL_TARGET_SIZE = 64


class EndToEndModel:
    def __init__(self):
        self.puzzle_localizer = HeuristicPuzzleLocalizer()
        self.digit_classifier = DigitClassifier()

    def __call__(self, inputs):
        """Parses a Sudoku image into a data structure.

        Arguments:
            inputs (tf.Tensor): A tensor of shape (H, W, C) where H and W are the height
                and width of the image respectively, and C is the number of channels.
                The channels are expected to be in the order RGB. The tensor values are
                expected to be floating-point numbers in the range [0, 1].

        Returns:
            tf.Tensor: A tensor of shape (9, 9) containing the predictions for the cells.
                The predictions are integers from 0 to 10 inclusive. See the `end_to_end`
                dataset readme for descriptions of the 11 classes. Dimension 0 of the
                output is the vertical axis, and dimension 1 is the horizontal axis.
        """
        assert inputs.ndim == 3, "Expected a 3D tensor of shape (H, W, C)"

        # Localize the puzzle
        # Scale by 255 since that's what the puzzle localizer has been tuned to.
        keypoints = self.puzzle_localizer(inputs * 255)

        # Make keypoints absolute
        keypoints = relative_keypoints_to_absolute(inputs, keypoints)

        # Unwarp the puzzle image
        puzzle_image = unwarp_puzzle_image(inputs.numpy(), keypoints)

        # Extract the cell images
        cells = extract_cells(puzzle_image)

        # Classify the cells
        predictions = np.zeros((9, 9), dtype=np.int32)
        for y in range(9):
            for x in range(9):
                pred = self.digit_classifier(cells[y][x]).numpy()
                predictions[y, x] = pred

        # TODO: Return tensor instead of numpy array, or make both input and output numpy arrays
        return predictions


def relative_keypoints_to_absolute(image, keypoints):
    # Scale the keypoints to be in terms of pixels
    keypoints_x = keypoints[0::2]
    keypoints_y = keypoints[1::2]
    keypoints_x = keypoints_x * image.shape[1]
    keypoints_y = keypoints_y * image.shape[0]
    keypoints = tf.reshape(tf.stack([keypoints_x, keypoints_y], axis=-1), (-1,))
    return keypoints


def unwarp_puzzle_image(image, keypoints):
    # Transform the puzzle image to face the camera
    matrix = cv.getPerspectiveTransform(
        np.reshape(keypoints, (4, 2)), PUZZLE_TARGET_KEYPOINTS
    )
    image = cv.warpPerspective(image, matrix, (PUZZLE_TARGET_SIZE, PUZZLE_TARGET_SIZE))
    return image


def extract_cells(puzzle_image):
    # Extract the cells
    return [
        [
            puzzle_image[
                y * CELL_TARGET_SIZE : (y + 1) * CELL_TARGET_SIZE,
                x * CELL_TARGET_SIZE : (x + 1) * CELL_TARGET_SIZE,
            ]
            for x in range(9)
        ]
        for y in range(9)
    ]
