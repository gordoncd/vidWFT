
## WaveTimeSeries Class

The [`WaveTimeSeries`](../vidWFT/core/process.py") class is designed to store and process wave time series data. It provides methods to clean raw position data, calculate power spectral density, and retrieve various wave properties.

### Initialization

```python
WaveTimeSeries(positions: np.ndarray, ppm: np.ndarray, sampling_rate: np.ndarray, data_is_raw: bool = False)
```

#### Parameters:
- **positions** (`np.ndarray`): Positions of the waves.
- **ppm** (`np.ndarray`): Pixels per meter.
- **sampling_rate** (`np.ndarray`): Sampling rate of the video.
- **data_is_raw** (`bool`, optional): Whether `positions` data is raw or not. Defaults to `False`.

### Methods

#### `clean_raw_positions`

```python
clean_raw_positions(raw_positions: np.ndarray, num_stakes: int, ppm: np.ndarray) -> np.ndarray
```

Cleans raw position data.

##### Parameters:
- **raw_positions** ([`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py)): Raw positions.
- **num_stakes** ([`int`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi)): Number of stakes.
- **ppm** ([`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py)): Pixels per meter.

##### Returns:
- [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py): Cleaned positions.

#### [`get_psd`](../vidWFT/core/process.py)

```python
get_psd() -> np.ndarray
```

Retrieves the power spectral density.

##### Returns:
- `np.ndarray`: Power spectral density.

#### `get_spec_freq`

```python
get_spec_freq() -> np.ndarray
```

Retrieves the spectral frequency.

##### Returns:
- [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py): Spectral frequency.

#### [`get_spec_ang_freq`](../vidWFT/core/process.py)

```python
get_spec_ang_freq() -> np.ndarray
```

Retrieves the spectral angular frequency.

##### Returns:
- `np.ndarray`: Spectral angular frequency.

#### `calc_psd`

```python
calc_psd(nfft: int = 2**10, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Calculates the power spectral density, frequency, and angular frequency using FFT.

##### Parameters:
- **nfft** ([`int`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi), optional): Number of points in FFT. Defaults to `2**10`.
- **kwargs**: Additional keyword arguments.
  - **fmin** ([`int`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi), optional): Minimum frequency to consider. Defaults to `None`.
  - **fmax** ([`int`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi), optional): Maximum frequency to consider. Defaults to `None`.

##### Returns:
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: Power spectral density, frequency, and angular frequency.

#### [`get_avg_wave_number`](../vidWFT/core/process.py)

```python
get_avg_wave_number(**kwargs) -> float
```

Retrieves the average wave number.

##### Returns:
- `float`: Average wave number.

#### `get_avg_wave_length`

```python
get_avg_wave_length() -> float
```

Retrieves the average wave length.

##### Returns:
- `float`: Average wave length.

#### [`get_avg_wave_height`](../vidWFT/core/process.py)

```python
get_avg_wave_height() -> float
```

Retrieves the average wave height.

##### Returns:
- `float`: Average wave height.

#### `get_avg_wave_period`

```python
get_avg_wave_period() -> float
```

Retrieves the average wave period.

##### Returns:
- `float`: Average wave period.

#### [`get_avg_wave_speed`](../vidWFT/core/process.py)

```python
get_avg_wave_speed() -> float
```

Retrieves the average wave speed.

##### Returns:
- `float`: Average wave speed.

#### `get_significant_wave_height`

```python
get_significant_wave_height() -> float
```

Retrieves the significant wave height.

##### Returns:
- `float`: Significant wave height.

