import os
import numpy as np 
import sys

# Add the parent directory of the tests directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now you can import the export module
from export import plot_wave_positions, extract_metadata_with_ffmpeg, write_metadata_to_txt, data_np_to_csv, prepare_files


def test_plot_wave_positions():
    arr = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
    path = 'test.png'
    plot_wave_positions(arr, path)
    assert os.path.exists(path)
    os.remove(path)

def test_extract_metadata_with_ffmpeg():
    metadata = extract_metadata_with_ffmpeg('/Users/gordondoore/Documents/GitHub/waves-summer-2024/videos/5k_perp_salmon.MP4')
    assert metadata['major_brand'] == 'mp41'
    assert metadata['minor_version'] == '538120216'
    assert metadata['compatible_brands'] == 'mp41'
    assert metadata['creation_time'] == '2024-07-03T19:47:04.000000Z'
    assert metadata['encoder'] == 'Lavf58.45.100'
    assert metadata['Duration'] == '00:19:02.11, start: 0.000000, bitrate: 55075 kb/s'
    assert metadata['Stream #0:0(eng)'] == 'Video: hevc (Main) (hvc1 / 0x31637668), yuvj420p(pc, bt709), 5312x2988 [SAR 1:1 DAR 16:9], 54818 kb/s, 29.97 fps, 29.97 tbr, 30k tbn, 29.97 tbc (default)'
    assert metadata['handler_name'] == 'GoPro MET'
    assert metadata['timecode'] == '15:47:04;07'
    
def test_write_metadata_to_txt():
    dest = 'test.txt'
    text_list = ['one','two','three']
    write_metadata_to_txt(dest, text_list)
    assert os.path.exists(dest)
    os.remove(dest)


def test_data_np_to_csv():
    data = np.array([[1,2,3],[4,5,6],[7,8,9]])
    headers = ['one','two','three']
    dest = 'test.csv'
    data_np_to_csv(data,dest, 'headers', headers = headers)
    assert os.path.exists(dest)
    os.remove(dest)


