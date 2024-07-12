'''
Gordon Doore
floats-video-analysis
07/10/2024

export.py
This file contains
functions for exporting files when floats video is extracted 

Last Updated: 07/10/2024

'''
import numpy as np 
import matplotlib.pyplot as plt
import subprocess
import os
import pandas as pd
import time


def plot_wave_positions(arr : np.ndarray, path : str) -> None:
    '''
    plot wave positions based on array of shape (T,num_stakes,2)
    '''
    num_stakes : int = arr.shape[1]
    fig : plt.Figure = plt.figure() #type: ignore
    for i in range(num_stakes):
        name : str = 'stake'+str(i)
        plt.plot(np.squeeze(np.where(arr[:,i,1]!=0)),label = name)
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.legend()
    fig.savefig(path)
    return

def extract_metadata_with_ffmpeg(video_path):
    '''
    Extract metadata from video using ffmpeg

    Args: 
        video_path (str): path to video file
    Returns: 
        metadata (dict): dictionary of metadata extracted from video
    '''
    # Run ffmpeg command to extract metadata
    try:
        cmd = ['ffmpeg', '-i', video_path, '-f', 'ffmetadata', '-']
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("Error executing ffmpeg command:", e)
        return None
    
    # ffmpeg writes metadata to stderr
    metadata_lines = result.stderr.split('\n')
    metadata = {}
    for line in metadata_lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            metadata[key.strip()] = value.strip()
    return metadata 

def write_metadata_to_txt(dest : str = 'output_metadata.txt', text_list : list[str] = []) -> None:
    if os.path.exists(dest):
        #ask for approval in command line to overwrite
        print('File already exists at '+dest)
        print('Do you want to overwrite this file?')
        print('Type "y" to overwrite, anything else will cancel')
        response = input()
        if response == 'n':
            return
        if response == 'y':
            print('Overwriting file...')
    with open(dest, 'w') as f:
        for line in text_list:
            f.write(line+'\n')
        f.close()

def generate_figures(raw_positions : np.ndarray, path : str) -> None:
    '''
    generate figures from raw positions array

    Args: 
        raw_positions (np.ndarray): array of raw positions
        path (str): path to save graph
    '''
    #TODO
    cleaned_positions = clean_raw_positions(raw_positions)
    #cleaned positions is an array of shape (T,num_stakes+1) containing only the y converted value of the positions and the time
    plot_wave_positions(cleaned_positions, path)
    #now we also want to generate spectrograms of each wave on the same figure
    #we will use the y values for this
    
    pass

def data_np_to_csv(data : np.ndarray, dest : str,header_string:str, index : bool = False,  **kwargs) -> None:
    '''
    convert numpy array to csv file

    Args: 
        data (np.ndarray): data to be converted
        dest (str): destination path for csv file
    '''
    #first we save the raw data
    df = pd.DataFrame(data)
    #now define the headers: 
    try:
        headers = kwargs.get(header_string, [str(i) for i in range(data.shape[1])])
    except Exception as e:
        print(e)
        print('Shape of data is: ',data.shape,' expected shape is (T,num_stakes*2)')
        print(dest+' did not save to csv')
        return
    if len(headers) != data.shape[1]:
        print('Error: headers must be the same length as the number of columns')
        print(dest+' did not save to csv')
        return
    
    try:
        df.columns = headers
    except Exception as e:
        print('Error: ',e)
        print('Make sure the raw data headers are the same length as the number of columns and use legal characters')
        print(dest+' did not save to csv')

    df.to_csv(dest,index = index)

def clean_raw_positions(raw_positions : np.ndarray, **kwargs) -> np.ndarray:
    '''
    clean raw positions array

    Args: 
        raw_positions (np.ndarray): array of raw positions
    '''
    #TODO
    #clean the raw positions array to remove zeros (unsampled timesteps)
    sampled_positions = raw_positions[np.where(raw_positions[:,:,1]!=0)]
    #now add a time column; just go by index for now 
    time = np.argwhere(raw_positions[:,0,1]!=0)
    
    #center around mean 



    return raw_positions

def assemble_text_output(video_path : str, calibration_data : tuple, ppm : np.ndarray, **kwargs) -> list[str]:
    '''
    assemble text output for output file

    Args: 
        video_path (str): path to video file
        metadata (dict): metadata extracted from video
        calibration_data (tuple): tuple of ndarrays (matrix_data, dist_data)
        ppm (np.ndarray): array of pixels per meter for each stake
    Returns:
        None    
    '''

    to_text = []
    to_text.append("This file contains the output data from the video to waveform conversion")
    to_text.append('_____________________________________________________________') 
    to_text.append('Date: '+time.strftime('%m/%d/%Y'))
    to_text.append('Time: '+time.strftime('%H:%M:%S'))
    to_text.append('_____________________________________________________________')
    to_text.append('Video Name: '+video_path)
    #extract metadata from video: 
    # Extract metadata from video using ffmpeg
    video_metadata = extract_metadata_with_ffmpeg(video_path)
    to_text.append('Video Metadata (Extracted with ffmpeg):')
    for key, value in video_metadata.items():
        to_text.append(f'{key}: {value}')
    to_text.append('_____________________________________________________________')

    num_stakes = ppm.shape[0]
    to_text.append('Number of Stakes: '+str(num_stakes)+'\n')
    mtx, dist = calibration_data
    to_text.append('Calibration Matrix: \n' + str(mtx)+'\n')
    to_text.append('Distance Coefficients: \n' + str(dist)+'\n')
    ppm_string = [f'stake_{i}: '+str(ppm[i]) for i in range(ppm.shape[0])]
    to_text.append('ppm for each stake: \n' + '\n'.join(ppm_string))
    to_text.append('_____________________________________________________________')

    #write kwargs to file 
    to_text.append('Additional Parameters: ')
    for key, value in kwargs.items():
        to_text.append(f'{key}: {value}')
    to_text.append('_____________________________________________________________')

    return to_text

def prepare_files(video_path : str, positions_data_raw: np.ndarray, calibration_data : tuple, ppm : np.ndarray, dest : str = '', **kwargs) -> None: 
    '''
    prepare output files from video to waveform

    Args: 
        video_path (str): path to unrectified video to be processed
        calibration_data (tuple): tuple of ndarrays (matrix_data, dist_data)
        dest (str): destination folder for output files

    Returns:
        None
    '''

    if dest == '':
        #make unique name for output based on input parameters
        dest = 'output_data/'+video_path.split('/')[-1].split('.')[0]+'_' #TODO: change this to something else
    if not os.path.exists(dest):
        os.makedirs(dest)
    graph_dest = kwargs.get('graph_dest', dest+'waveform_graph.png')
    raw_csv_dest = kwargs.get('raw_csv_dest', dest+'positions_raw.csv')
    clean_csv_dest = kwargs.get('clean_csv_dest', dest+'positions_clean.csv')
    txt_dest = kwargs.get('txt_dest', dest+'metadata.txt')

        
    #now generate text to put to the txt file
    #the file will include the video name, any metadata about that video, calibration data, the ppm for each stake
    to_text = assemble_text_output(video_path, calibration_data, ppm, **kwargs)

    # floats_video_to_waveform('videos/noodle_float_move_rect.mp4',750,2)
    #now save the text to a file at txt_dest
    write_metadata_to_txt(txt_dest, to_text)

    #we now convert waveform arrays to csv files with headers
    #we will save these to csv_dest
    #first we save the raw data
    data_np_to_csv(positions_data_raw, raw_csv_dest, 'raw_headers', index = True, **kwargs)
    #now save the cleaned data
    positions_data_clean = clean_raw_positions(positions_data_raw, **kwargs)
    data_np_to_csv(positions_data_clean, clean_csv_dest, 'cleaned_headers', index = False,**kwargs)

    #now we save the graph to graph_dest
    generate_figures(positions_data_raw, graph_dest)
    return
