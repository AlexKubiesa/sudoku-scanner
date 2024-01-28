# Puzzle localization dataset

This dataset consists of 80 train, validation and test images of Sudoku puzzles (33:33:33 split). They are the same images as in the `end_to_end` dataset.

Each image is labelled with four (x, y) keypoints, which are the positions of the corners of the Sudoku puzzle. The origin (0, 0) is in the top-left of the image.

The task for this dataset is to correctly determine the locations of the Sudoku puzzle corners.

## Labels file format

The validation and test subsets each have a file called `labels.csv` containing metadata and image labels. The file should be the same as `metadata.csv` from the `end_to_end` dataset. The columns are as follows:
* **Name**: The image file name.
* **Source**: Specifies how the image was obtained. Possible values:
  * `honor_view_10`: Photo taken from Honor View 10 phone.
  * `iphone_14`: Photo taken from iPhone 14.
* **Puzzle**: A date of the form `MM_DD`, where `MM` is the two-digit month and `DD` is the two-digit day. The date identifies the Sudoku puzzle. Each puzzle corresponds to a day of the year in 2022, except weekend puzzles count for both Saturday and Sunday. For weekend puzzles, the date corresponds to the Saturday.
* **Width**: The width of the image in pixels.
* **Height**: The height of the image in pixels.
* **x_topleft**, **y_topleft**, ..., **x_topright**, **y_topright**: The (x, y) coordinates of the top-left, bottom-left, bottom-right and top-right corners of the Sudoku puzzle, relative to the width and height of the full image. (0, 0) is the top-left corner of the image.
* **x_topleft_absolute**, ..., **y_topright_absolute**: The (x, y) coordinates of the corners of the Sudoku grid, given in pixels.
