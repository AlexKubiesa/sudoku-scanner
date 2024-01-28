# End-to-end dataset

This dataset consists of 80 train, validation and test images of Sudoku puzzles (33:33:33 split). The Sudoku puzzles are partially or fully completed, so consists of the original printed digits as well as handwritten digits. Some cells contain multiple digits, which are candidates for the given cell.

Each image is labelled with the set of numbers written in each cell. The (x, y) positions of the Sudoku puzzle corners are also given as metadata.

The task for this dataset is to extract the Sudoku puzzle and correctly determine the state of all the cells. The suggested target is 98% accuracy.

## Composition

The train set consists of photos of 20 Sudoku puzzles. For each puzzle, the following photos were taken:
  * Honor View 10 phone
    * One portrait (3456 width, 4608 height)
    * One landscape (4608 width, 3456 height)
  * iPhone 14
    * One portrait (3024 width, 4032 height)
    * One landscape (4032 width, 3024 height)

This makes a total of 20 x 2 x 2 = 80 photos.

The val and test sets consist of the same breakdown of photos, but of different Sudoku puzzles, to minimize the correlations between the different subsets.

## Labels file format

Each image `images/name.jpg` has a corresponding labels file `labels/name.txt`. The labels file contains 9 rows of 9 integers each, separated by spaces. The labels are laid out in the same shape as the Sudoku grid. The file can easily be read into a numpy array with `np.loadtxt`.

There are 11 classes:
* `0` (blank): No digits in cell.
* `1-9`: One digit in cell (which is not just a candidate).
* `candidates`: One or more candidates in cell.

## Metadata file format

The validation and test subsets each have a file called `metadata.csv` containing metadata for the images. The columns are as follows:
* **Name**: The image file name.
* **Source**: Specifies how the image was obtained. Possible values:
  * `honor_view_10`: Photo taken from Honor View 10 phone.
  * `iphone_14`: Photo taken from iPhone 14.
* **Puzzle**: A date of the form `MM_DD`, where `MM` is the two-digit month and `DD` is the two-digit day. The date identifies the Sudoku puzzle. Each puzzle corresponds to a day of the year in 2022, except weekend puzzles count for both Saturday and Sunday. For weekend puzzles, the date corresponds to the Saturday.
* **Width**: The width of the image in pixels.
* **Height**: The height of the image in pixels.
* **x_topleft**, **y_topleft**, ..., **x_topright**, **y_topright**: The (x, y) coordinates of the top-left, bottom-left, bottom-right and top-right corners of the Sudoku puzzle, relative to the width and height of the full image. (0, 0) is the top-left corner of the image.
* **x_topleft_absolute**, ..., **y_topright_absolute**: The (x, y) coordinates of the corners of the Sudoku grid, given in pixels.
