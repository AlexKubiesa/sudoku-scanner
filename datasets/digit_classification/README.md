# Digit classification dataset

This dataset consists of 6480 train, validation and test images of cells extracted from a Sudoku grid (33:33:33 split). The cell images are extracted from the puzzle images in the `end_to_end` dataset using the puzzle keypoints.

There are 11 classes:
* `0` (blank): No digits in cell.
* `1-9`: One digit in cell (which is not just a candidate).
* `candidates`: One or more candidates in cell.

The task for this dataset is to label the cell with the correct class. The suggested target is 99% accuracy.
