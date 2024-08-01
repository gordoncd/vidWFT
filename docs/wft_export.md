
### [`export.py`](../vidWFT/core/export.py) Documentation

# Export Module

This module contains functions for exporting files when floats video is extracted. It includes functions for plotting wave positions, extracting metadata from videos, writing metadata to text files, generating figures, converting numpy arrays to CSV files, cleaning raw positions, and preparing output files.

## Functions

### [`plot_wave_positions`](../vidWFT/core/export.py)

```python
def plot_wave_positions(arr: np.ndarray, path: str) -> None:
    """
    Plot wave positions based on an array of shape (T, num_stakes, 2).

    Args:
        arr (np.ndarray): Array containing wave positions.
        path (str): Path to save the plot.

    Returns:
        None
    """
```

#### Description
This function plots wave positions based on the provided array and saves the plot to the specified path.

#### Parameters
- `arr` (np.ndarray): Array of shape (T, num_stakes, 2) containing wave positions.
- `path` (str): Path to save the plot.

#### Returns
- `None`

### `extract_metadata_with_ffmpeg`

```python
def extract_metadata_with_ffmpeg(video_path: str) -> dict:
    """
    Extract metadata from video using ffmpeg.

    Args:
        video_path (str): Path to video file.

    Returns:
        dict: Dictionary of metadata extracted from video.
    """
```

#### Description
This function extracts metadata from a video file using the `ffmpeg` command-line tool.

#### Parameters
- [`video_path`]( ../vidWFT/core/export.py) (str): Path to the video file.

#### Returns
- `dict`: Dictionary containing the extracted metadata.

### [`write_metadata_to_txt`](../vidWFT/core/export.py)

```python
def write_metadata_to_txt(dest: str = 'output_metadata.txt', text_list: list[str] = []) -> None:
    """
    Write metadata to a text file.

    Args:
        dest (str): Destination path for the text file.
        text_list (list[str]): List of strings to write to the text file.

    Returns:
        None
    """
```

#### Description
This function writes a list of strings to a text file. If the file already exists, it asks for user approval to overwrite it.

#### Parameters
- `dest` (str): Destination path for the text file. Default is 'output_metadata.txt'.
- `text_list` (list[str]): List of strings to write to the text file.

#### Returns
- `None`

### `generate_figures`

```python
def generate_figures(raw_positions: np.ndarray, path: str) -> None:
    """
    Generate figures from raw positions array.

    Args:
        raw_positions (np.ndarray): Array of raw positions.
        path (str): Path to save the graph.

    Returns:
        None
    """
```

#### Description
This function generates figures from a raw positions array and saves them to the specified path.

#### Parameters
- [`raw_positions`](../vidWFT/core/export.py) (np.ndarray): Array of raw positions.
- [`path`](../vidWFT/core/export.py) (str): Path to save the graph.

#### Returns
- `None`

### [`data_np_to_csv`](../vidWFT/core/export.py)

```python
def data_np_to_csv(data: np.ndarray, dest: str, header_string: str, index: bool = False, **kwargs) -> None:
    """
    Convert numpy array to CSV file.

    Args:
        data (np.ndarray): Data to be converted.
        dest (str): Destination path for the CSV file.
        header_string (str): Header string for the CSV file.
        index (bool): Whether to include the index in the CSV file. Default is False.

    Returns:
        None
    """
```

#### Description
This function converts a numpy array to a CSV file and saves it to the specified destination.

#### Parameters
- `data` (np.ndarray): Data to be converted.
- `dest` (str): Destination path for the CSV file.
- `header_string` (str): Header string for the CSV file.
- `index` (bool): Whether to include the index in the CSV file. Default is False.

#### Returns
- `None`

### `clean_raw_positions`

```python
def clean_raw_positions(raw_positions: np.ndarray, **kwargs) -> np.ndarray:
    """
    Clean raw positions array.

    Args:
        raw_positions (np.ndarray): Array of raw positions.

    Returns:
        np.ndarray: Cleaned positions array.
    """
```

#### Description
This function cleans a raw positions array by removing zeros (unsampled timesteps) and adding a time column.

#### Parameters
- [`raw_positions`](../vidWFT/core/export.py) (np.ndarray): Array of raw positions.

#### Returns
- [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py): Cleaned positions array.

### [`assemble_text_output`](../vidWFT/core/export.py)

```python
def assemble_text_output(video_path: str, calibration_data: tuple, ppm: np.ndarray, **kwargs) -> list[str]:
    """
    Assemble text output for output file.

    Args:
        video_path (str): Path to video file.
        calibration_data (tuple): Tuple of ndarrays (matrix_data, dist_data).
        ppm (np.ndarray): Array of pixels per meter for each stake.

    Returns:
        list[str]: List of strings to write to the text file.
    """
```

#### Description
This function assembles text output for an output file, including video metadata, calibration data, and additional parameters.

#### Parameters
- `video_path` (str): Path to the video file.
- `calibration_data` (tuple): Tuple of ndarrays (matrix_data, dist_data).
- `ppm` (np.ndarray): Array of pixels per meter for each stake.

#### Returns
- `list[str]`: List of strings to write to the text file.

### `prepare_files`

```python
def prepare_files(video_path: str, positions_data_raw: np.ndarray, calibration_data: tuple, ppm: np.ndarray, dest: str = '', **kwargs) -> None:
    """
    Prepare output files from video to waveform.

    Args:
        video_path (str): Path to unrectified video to be processed.
        positions_data_raw (np.ndarray): Raw positions data.
        calibration_data (tuple): Tuple of ndarrays (matrix_data, dist_data).
        ppm (np.ndarray): Array of pixels per meter for each stake.
        dest (str): Destination folder for output files. Default is ''.

    Returns:
        None
    """
```

#### Description
This function prepares output files from video to waveform, including text files, CSV files, and graphs.

#### Parameters
- [`video_path`](../vidWFT/core/export.py) (str): Path to the unrectified video to be processed.
- [`positions_data_raw`](../vidWFT/core/export.py) (np.ndarray): Raw positions data.
- [`calibration_data`](../vidWFT/core/export.py) (tuple): Tuple of ndarrays (matrix_data, dist_data).
- [`ppm`](../vidWFT/core/export.py) (np.ndarray): Array of pixels per meter for each stake.
- [`dest`](../vidWFT/core/export.py) (str): Destination folder for output files. Default is ''.

#### Returns
- `None`

## Usage Examples

### Example 1: Plotting Wave Positions

```python
import numpy as np
from export import plot_wave_positions

# Example data
arr = np.random.rand(100, 5, 2)
path = 'wave_positions.png'

# Plot wave positions
plot_wave_positions(arr, path)
```

### Example 2: Extracting Metadata from Video

```python
from export import extract_metadata_with_ffmpeg

# Example video path
video_path = 'example_video.mp4'

# Extract metadata
metadata = extract_metadata_with_ffmpeg(video_path)
print(metadata)
```

### Example 3: Writing Metadata to Text File

```python
from export import write_metadata_to_txt

# Example text list
text_list = ['Line 1', 'Line 2', 'Line 3']
dest = 'output_metadata.txt'

# Write metadata to text file
write_metadata_to_txt(dest, text_list)
```

### Example 4: Generating Figures

```python
import numpy as np
from export import generate_figures

# Example data
raw_positions = np.random.rand(100, 5, 2)
path = 'figures.png'

# Generate figures
generate_figures(raw_positions, path)
```

### Example 5: Converting Numpy Array to CSV

```python
import numpy as np
from export import data_np_to_csv

# Example data
data = np.random.rand(100, 10)
dest = 'output.csv'
header_string = 'header'

# Convert numpy array to CSV
data_np_to_csv(data, dest, header_string)
```

### Example 6: Cleaning Raw Positions

```python
import numpy as np
from export import clean_raw_positions

# Example data
raw_positions = np.random.rand(100, 5, 2)

# Clean raw positions
cleaned_positions = clean_raw_positions(raw_positions)
print(cleaned_positions)
```

### Example 7: Preparing Files

```python
import numpy as np
from export import prepare_files

# Example data
video_path = 'example_video.mp4'
positions_data_raw = np.random.rand(100, 5, 2)
calibration_data = (np.random.rand(3, 3), np.random.rand(5))
ppm = np.random.rand(5)
dest = 'output_folder/'

# Prepare files
prepare_files(video_path, positions_data_raw, calibration_data, ppm, dest)
```