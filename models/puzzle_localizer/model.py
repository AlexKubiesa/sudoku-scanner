from pathlib import Path
import logging

import PIL.Image
import PIL.ImageOps
import PIL.ImageFilter
import tensorflow as tf
import numpy as np
import cv2 as cv


EPSILON = 1e-8


class HeuristicPuzzleLocalizer:
    def __init__(self, blur_size=40, threshold=0.1):
        self.blur_size = blur_size
        self.threshold = threshold

    def __call__(self, inputs):
        """Localizes a Sudoku puzzle in an image.

        Arguments:
            inputs (tf.Tensor): A tensor of shape (H, W, C) where H and W are the height
                and width of the image respectively, and C is the number of channels.
                The channels are expected to be in the order RGB. The tensor values are
                expected to be in the range [0, 255].

        Returns:
            tf.Tensor: A tensor of shape (8,) containing the x and y coordinates
                of the top-left, bottom-left, bottom-right, top-right corners of the
                puzzle, in that order.
        """
        assert inputs.ndim in [3, 4], "inputs must be 3D or 4D"
        is_batched = inputs.ndim == 4

        if not is_batched:
            inputs = tf.expand_dims(inputs, axis=0)

        logging.info("Localizing puzzle batch.")
        _, height, width, _ = inputs.shape

        # Convert images to grayscale
        inputs = tf.image.rgb_to_grayscale(inputs)

        # Binarize images with adaptive thresholding
        inputs = adaptive_threshold(
            inputs, blur_size=self.blur_size, threshold=self.threshold
        )

        outputs = []

        for image in inputs:
            # Find the largest component according to some criteria
            component = get_largest_connected_component(
                image,
                ConnectedComponentCriteria(
                    min_size=min(height, width) * 0.3,
                    max_size=min(height, width) * 0.9,
                    min_aspect_ratio=0.5,
                    max_aspect_ratio=1.5,
                ),
            )

            # If no suitable component was found, just output zeros
            if component is None:
                outputs.append(tf.zeros((8,), dtype=tf.float32))
                continue

            # Get corner points
            (
                (x_topleft, y_topleft),
                (x_bottomleft, y_bottomleft),
                (x_bottomright, y_bottomright),
                (x_topright, y_topright),
            ) = get_corner_points(component)

            # Scale corner points to image size
            x_topleft /= width
            x_bottomleft /= width
            x_bottomright /= width
            x_topright /= width

            y_topleft /= height
            y_bottomleft /= height
            y_bottomright /= height
            y_topright /= height

            outputs.append(
                tf.constant(
                    (
                        x_topleft,
                        y_topleft,
                        x_bottomleft,
                        y_bottomleft,
                        x_bottomright,
                        y_bottomright,
                        x_topright,
                        y_topright,
                    ),
                    dtype=tf.float32,
                )
            )

        outputs = tf.stack(outputs)

        if not is_batched:
            outputs = outputs[0]

        return outputs


def adaptive_threshold(images, blur_size, threshold):
    blurred = box_blur(images, blur_size)
    return tf.cast(blurred - images > threshold, tf.int8)


def box_blur(images, kernel_size):
    channels = images.shape[-1]
    assert channels == 1, "Only grayscale images are supported"
    filter = tf.ones((kernel_size, kernel_size, channels, 1)) / float(kernel_size**2)
    images = tf.nn.conv2d(images, filter, [1, 1, 1, 1], padding="SAME")
    return images


def get_largest_connected_component(binary_image, criteria):
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(
        binary_image.numpy(), connectivity=4
    )

    best_index = None
    best_area = float("-inf")

    # stats has 5 columns: x_left, y_top, width, height, area
    num_components = stats.shape[0]
    for i in range(num_components):
        x_left, y_top, width, height, area = stats[i]
        aspect_ratio = width / (height + EPSILON)
        if (
            aspect_ratio >= criteria.min_aspect_ratio
            and aspect_ratio <= criteria.max_aspect_ratio
            and height >= criteria.min_size
            and height <= criteria.max_size
            and width >= criteria.min_size
            and width <= criteria.max_size
        ):
            if best_index is None or area > best_area:
                logging.debug(
                    "Found new best component. index = %s, (x_left, y_top) = (%s, %s), width = %s, height = %s, area = %s, ",
                    i,
                    x_left,
                    y_top,
                    width,
                    height,
                    area,
                )
                best_index = i
                best_area = area

    if best_index is None:
        return None

    logging.debug("Best index = %s", best_index)
    logging.debug("stats[best_index] = %s", stats[best_index])
    x_left, y_top, width, height, area = stats[best_index]
    bounds = ((x_left, y_top), (x_left + width, y_top + height))

    points = []
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] == best_index:
                points.append((x, y))

    logging.debug("Bounds: %s", bounds)
    logging.debug("Points: %s ...", points[:10])

    return ConnectedRegion(points, bounds)


class ConnectedRegion:
    def __init__(self, points, bounds):
        assert isinstance(points, list), "points must be a list"
        assert len(points) > 0, "points must not be empty"
        self.points = points
        self.bounds = bounds

    def width(self):
        ((x_min, y_min), (x_max, y_max)) = self.bounds
        return x_max - x_min

    def height(self):
        ((x_min, y_min), (x_max, y_max)) = self.bounds
        return y_max - y_min

    def aspect_ratio(self):
        return self.width() / (self.height() + EPSILON)

    def bounding_box_area(self):
        return self.width() * self.height()


class ConnectedComponentCriteria:
    def __init__(self, min_size, max_size, min_aspect_ratio, max_aspect_ratio):
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio


def get_corner_points(region):
    (x_min, y_min), (x_max, y_max) = region.bounds
    points = region.points
    top_left = get_nearest_point(points, x_min, y_min)
    bottom_left = get_nearest_point(points, x_min, y_max)
    bottom_right = get_nearest_point(points, x_max, y_max)
    top_right = get_nearest_point(points, x_max, y_min)
    return top_left, bottom_left, bottom_right, top_right


def get_nearest_point(points, x, y):
    best_point = None
    best_distance = float("inf")

    for point in points:
        dx = abs(point[0] - x)
        dy = abs(point[1] - y)
        distance = dx + dy

        if distance < best_distance:
            best_point = point
            best_distance = distance

    return best_point
