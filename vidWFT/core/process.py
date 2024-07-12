'''
Author: Gordon Doore 
Created: 07/12/2024
process.py
data processing functions

Last Updated: 07/12/2024
'''

import oceanlyz as oz
import numpy as np

class WaveTimeSeries:
    '''
    object to store wave time series data

    '''
    def __init__(self, positions: np.ndarray, ppm: np.ndarray, sampling_rate: np.ndarray, data_is_raw: bool = False):
        '''
        Initialize the Process class.

        Args:
            positions (np.ndarray): Positions of the waves.
            ppm (np.ndarray): Pixels per meter.
            sampling_rate (np.ndarray): Sampling rate of the video.
            data_is_raw (bool, optional): Whether `positions` data is raw or not. Defaults to False.
        '''

        if data_is_raw: 
            self.positions = self.clean_raw_positions(positions, positions.shape[1], ppm)
        self.positions = positions
        self.ppm = ppm
        self.sampling_rate = sampling_rate

    def clean_raw_positions(self, raw_positions : np.ndarray, num_stakes : int, ppm: np.ndarray) -> np.ndarray:
        '''
        clean raw positions

        Args:
            raw_positions (np.ndarray): raw positions
            num_stakes (int): number of stakes
        Returns:
            np.ndarray: cleaned positions
        '''
        #positions are taken every x frames, so we want to remove the gaps 
        sampled = raw_positions[np.where(raw_positions[:,:,1]!=0)]
        positions = np.zeros((sampled.shape[0],num_stakes+1))
        positions[:,1:]= np.reshape(sampled[:,:,1],(sampled.shape[0],num_stakes))
        #now we want to add the time column based on the index in raw_positions
        positions[:,0] = np.argwhere(raw_positions[:,0,1]!=0).flatten()
        
        #now we want to convert the y values to real values based on ppm 
        positions[:,1:] = positions[:,1:] / ppm
        #now we center the positions around the mean of each column
        positions[:,1:] = positions[:,1:] - np.mean(positions[:,1:],axis = 0)
        
        return positions

    def get_psd(self):
        '''
        get power spectral density
        '''


        pass

    def get_avg_wave_number(self):
        '''
        get wave number
        kwargs define method of analysis
        '''
        pass

    def get_avg_wave_length(self):
        '''
        get wave length
        kwargs define method of analysis
        '''
        pass

    def get_avg_wave_height(self):
        '''
        get wave height
        kwargs define method of analysis
        '''
        pass

    def get_avg_wave_period(self):
        '''
        get wave period
        kwargs define method of analysis
        '''
        pass

    def get_avg_wave_speed(self):
        '''
        get wave speed
        kwargs define method of analysis
        '''
        pass

    def get_significant_wave_height(self):
        '''
        get significant wave height
        kwargs define method of analysis
        '''
        pass





