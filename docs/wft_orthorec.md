# Documentation for [`orthorec.py`](../vidWFT/core/orthorec.py)

## Overview

This module provides functions for image rectification and processing based on user-defined points and gradations. It includes functionalities for picking points on an image, rectifying images based on perspective transformations, and finding gradations in images.

## Functions

### [`pick_points`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2FUsers%2Fgordondoore%2FDocuments%2FGitHub%2FvidWFT%2FvidWFT%2Fcore%2Forthorec.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A18%2C%22character%22%3A4%7D%5D ../vidWFT/core/orthorec.py)

```python
def pick_points(img: np.ndarray) -> list[Any]:
```

Allows the user to interactively pick points on an image.

- **Parameters:**
  - `img` (np.ndarray): The input image on which points are to be picked.

- **Returns:**
  - `list[Any]`: A list of points picked by the user.

- **Usage:**
  ```python
  points = pick_points(image)
  ```

### `rectify`

```python
def rectify(img: np.ndarray, inpoints: np.ndarray, outpoints: np.ndarray) -> np.ndarray:
```

Rectifies an image based on input and output points using a perspective transformation.

- **Parameters:**
  - [`img`](../vidWFT/core/orthorec.py) (np.ndarray): The input image to be rectified.
  - [`inpoints`](../vidWFT/core/orthorec.py) (np.ndarray): The original points in the image.
  - [`outpoints`](../vidWFT/core/orthorec.py) (np.ndarray): The points to map to.

- **Returns:**
  - [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): The rectified image.

- **Usage:**
  ```python
  rectified_image = rectify(image, original_points, destination_points)
  ```

### `order_points`

```python
def order_points(pts: np.ndarray) -> np.ndarray:
```

Orders points in a consistent manner: top-left, top-right, bottom-right, bottom-left.

- **Parameters:**
  - [`pts`](../vidWFT/core/orthorec.py) (np.ndarray): Array of points to be ordered.

- **Returns:**
  - [`np.ndarray`]( ../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): Ordered points.

- **Usage:**
  ```python
  ordered_points = order_points(points)
  ```

### `two_largest`

```python
def two_largest(arr: np.ndarray) -> Tuple[list[Any], list[Any]]:
```

Finds the two largest consecutive subsequences in an array.

- **Parameters:**
  - [`arr`](../vidWFT/core/orthorec.py) (np.ndarray): Input array.

- **Returns:**
  - [`Tuple[list[Any], list[Any]]`](../../../../opt/anaconda3/lib/python3.9/typing.py"): Two largest consecutive subsequences.

- **Usage:**
  ```python
  first, second = two_largest(array)
  ```

### `find_difference_gradations`

```python
def find_difference_gradations(gradation_pix: Iterable[np.ndarray]):
```

Finds the difference between the centers of the two largest gradations in each column.

- **Parameters:**
  - [`gradation_pix`](../vidWFT/core/orthorec.py) (Iterable[np.ndarray]): Iterable of pixel columns.

- **Returns:**
  - [`list[float]`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi"): List of distances between the centers of the two largest gradations.

- **Usage:**
  ```python
  distances = find_difference_gradations(gradation_pixels)
  ```

### `find_gradations`

```python
def find_gradations(img: np.ndarray, lines: list[np.ndarray], threshold_condition: Callable[[np.ndarray], np.ndarray]):
```

Finds the gradations in an image based on user-defined lines and a threshold condition.

- **Parameters:**
  - [`img`](../vidWFT/core/orthorec.py) (np.ndarray): Input image.
  - [`lines`](../vidWFT/core/orthorec.py) (list[np.ndarray]): List of lines defined by the user.
  - [`threshold_condition`](../vidWFT/core/orthorec.py) (Callable[[np.ndarray], np.ndarray]): Function to threshold pixel values.

- **Returns:**
  - [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): Points of the center of the gradations.

- **Usage:**
  ```python
  gradation_points = find_gradations(image, lines, threshold_condition)
  ```

### `get_ppm`

```python
def get_ppm(img: np.ndarray, points: np.ndarray, pixel_columns: Iterable[np.ndarray], gradation_size: np.ndarray, stake_thresh: int, stake_grad_thresh: int, peaks_sampled: int) -> np.ndarray:
```

Calculates pixels per millimeter (ppm) for different stakes in an image.

- **Parameters:**
  - [`img`](../vidWFT/core/orthorec.py) (np.ndarray): Input image.
  - [`points`](../vidWFT/core/orthorec.py) (np.ndarray): Points of gradations.
  - [`pixel_columns`](../vidWFT/core/orthorec.py) (Iterable[np.ndarray]): Columns of pixel values.
  - [`gradation_size`](../vidWFT/core/orthorec.py) (np.ndarray): Size of gradations.
  - [`stake_thresh`](../vidWFT/core/orthorec.py) (int): Threshold for stakes.
  - [`stake_grad_thresh`](../vidWFT/core/orthorec.py) (int): Threshold for gradations.
  - [`peaks_sampled`](../vidWFT/core/orthorec.py) (int): Number of peaks sampled.

- **Returns:**
  - [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): Pixels per millimeter for each stake.

- **Usage:**
  ```python
  ppm_values = get_ppm(image, points, pixel_columns, gradation_size, stake_thresh, stake_grad_thresh, peaks_sampled)
  ```

### `linear_transform`

```python
def linear_transform(points: np.ndarray):
```

Calculates the linear transformation for perspective warping between two stakes with gradations.

- **Parameters:**
  - [`points`](../vidWFT/core/orthorec.py) (np.ndarray): Array of points with shape (N_stakes, 2, 2).

- **Returns:**
  - [`Tuple[np.ndarray, np.ndarray]`](../../../../opt/anaconda3/lib/python3.9/typing.py"): Slopes and intercepts for each stake.

- **Usage:**
  ```python
  slopes, intercepts = linear_transform(points)
  ```

### `rectify_by_gradation`

```python
def rectify_by_gradation(img: np.ndarray, n_stakes: int, stake_thresh: int, stake_grad_thresh: int, threshold_condition: Callable[[np.ndarray], np.ndarray], load_prev_grad: np.ndarray = np.zeros((4,))) -> Tuple[np.ndarray, np.ndarray]:
```

Rectifies an image by size variation based on gradations.

- **Parameters:**
  - [`img`](../vidWFT/core/orthorec.py) (np.ndarray): Input image.
  - [`n_stakes`](../vidWFT/core/orthorec.py) (int): Number of stakes.
  - [`stake_thresh`](../vidWFT/core/orthorec.py) (int): Threshold for stakes.
  - [`stake_grad_thresh`](../vidWFT/core/orthorec.py) (int): Threshold for gradations.
  - [`threshold_condition`](../vidWFT/core/orthorec.py) (Callable[[np.ndarray], np.ndarray]): Function to threshold pixel values.
  - [`load_prev_grad`](../vidWFT/core/orthorec.py) (np.ndarray): Previous gradation points.

- **Returns:**
  - [`Tuple[np.ndarray, np.ndarray]`](../../../../opt/anaconda3/lib/python3.9/typing.py"): Rectified image and old points.

- **Usage:**
  ```python
  rectified_image, old_points = rectify_by_gradation(image, n_stakes, stake_thresh, stake_grad_thresh, threshold_condition)
  ```

### `define_stakes`

```python
def define_stakes(img: np.ndarray, n_stakes: int) -> Tuple[list[list[Any]], list[Any]]:
```

Allows the user to draw lines on an image to define stakes.

- **Parameters:**
  - [`img`](../vidWFT/core/orthorec.py) (np.ndarray): Input image.
  - [`n_stakes`](../vidWFT/core/orthorec.py) (int): Number of stakes.

- **Returns:**
  - [`Tuple[list[list[Any]], list[Any]]`](../../../../opt/anaconda3/lib/python3.9/typing.py"): Coordinates for each stake and lines computed using [`skimage.draw.line`](../../../../opt/anaconda3/lib/python3.9/site-packages/skimage/__init__.py").

- **Usage:**
  ```python
  points, lines = define_stakes(image, n_stakes)
  ```

## File Location

This module is located at `../vidWFT/core/orthorec.py`.