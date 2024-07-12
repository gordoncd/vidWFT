'''
Author: Gordon Doore 
Created: 07/12/2024
process.py
data processing functions

Last Updated: 07/12/2024
'''

import oceanlyz as oz
import numpy as np
import scipy as sp

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

        # Check if the data is raw and needs cleaning
        if data_is_raw: 
            self.positions = self.clean_raw_positions(positions, positions.shape[1], ppm)
        
        # Store the positions, pixels per meter, and sampling rate
        self.positions = positions
        self.ppm = ppm
        self.sampling_rate = sampling_rate
        
        # Initialize variables for power spectral density and frequencies
        self.psd = None
        self.spec_freq = None
        self.spec_ang_freq = None
        
        # Initialize variables for average wave properties
        self.avg_wave_height = None
        self.avg_wave_length = None
        self.avg_wave_number = None
        self.avg_wave_period = None
        self.avg_wave_speed = None
        self.significant_wave_height = None


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
        return self.psd
    
    def get_spec_freq(self):
        '''
        get spectral frequency
        '''
        return self.spec_freq
    
    def get_spec_ang_freq(self):
        '''
        get spectral angular frequency
        '''
        return self.spec_ang_freq
    
    def calc_psd(self, nfft: int = 2^10, **kwargs):
        '''
        calculate power spectral density -- also calcs frequency and angular frequency (from fft)

        function derived from oceanlyz package -- this is essentially a refactoring of that code
        to fit the class structure and to allow for more flexibility in the future
        
        oceanlyz.Python Functions/WaveSpectraFun.py: 
        https://github.com/akarimp/Oceanlyz/blob/master/Python%20Functions/WaveSpectraFun.py 


        keyword arguments:
            fmin (int, optional): minimum frequency to consider. Defaults to None.
            fmax (int, optional): maximum frequency to consider. Defaults to None.
        '''

        f,Syy=sp.signal.welch(self.positions,fs=self.sampling_rate,nfft=nfft) #Wave power spectrum and Frequency
        w = 2*np.pi*f #Angular frequency
        self.psd = Syy
        self.spec_freq = f
        self.spec_ang_freq = w

        #setting syy min and max
        fmin = kwargs.get('fmin',None)
        if fmin is not None:
            Syy[f<fmin] = 0
        fmax = kwargs.get('fmax',None)
        if fmax is not None:
            Syy[f>fmax] = 0
        
        return Syy, f, w

    def get_avg_wave_number(self, **kwargs):
        '''
        get wave number
        '''


        return self.avg_wave_number

    def get_avg_wave_length(self):
        '''
        get wave length
        '''
        return self.avg_wave_length

    def get_avg_wave_height(self):
        '''
        get wave height
        '''
        return self.avg_wave_height

    def get_avg_wave_period(self):
        '''
        get wave period
        '''
        return self.avg_wave_period

    def get_avg_wave_speed(self):
        '''
        get wave speed
        '''
        return self.avg_wave_speed

    def get_significant_wave_height(self):
        '''
        get significant wave height
        '''
        return self.significant_wave_height


   


