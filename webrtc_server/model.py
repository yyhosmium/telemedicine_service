import sys
import torch
from models.QrPPGNet_v3 import *
from training_functions import EarlyStopping, SkinColorFilter, Neg_Pearson
from face_tracking import SignalProcessing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import butter,detrend,welch
from scipy.fft import fft, fftfreq
import os
from torch import nn
import time


def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(float32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

class BPM:
    """
    Provides BPMs estimate from BVP signals using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz


    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        Pmax = np.argmax(Power, axis=1)  # power max
        SNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        Power = (Power-np.min(Power))/(np.max(Power)-np.min(Power))
        pSNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        return Pfreqs[Pmax.squeeze()], SNR, pSNR, Pfreqs, Power



image_dim = 128
T = 64+1 #첫프레임 버리기

def process_video(video_frames):
    global image_dim, T
    processed_frames = np.zeros((T,image_dim,image_dim,3)) #미리 배열생성
    frames_idx = 0
    signal_processor = SignalProcessing(video_frames)

    for frame in signal_processor.extract_holistic(face_detection_interval=1): #face tracking 결과 return
        frame = cv2.resize(frame.astype(np.uint8),dsize=(image_dim,image_dim),interpolation=cv2.INTER_AREA)
        processed_frames[frames_idx] = frame
        frames_idx +=1
    processed_frames = processed_frames[1:]
    return processed_frames


def find_max_magnitude_hz(data,sampling_rate=30, low_band=0.5, high_band=2.5):
    N=len(data)
    T=1.0/sampling_rate
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = data
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    yf = 2.0/N*np.abs(yf[0:N//2])
    x_f_in_range_idx = np.where((xf>=low_band)&(xf<=high_band))
    xf = xf[x_f_in_range_idx]
    yf = yf[x_f_in_range_idx]
    max_PS_idx = np.argmax(yf)
    return xf[max_PS_idx]


def model(video_path):
    global T
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    width, height = 640, 480
    input = np.zeros((T,height,width,3), dtype=np.uint8)
    
    if cap.isOpened():
        for i in range(0,T):
            ret, frame = cap.read()        
            if not ret:
                break
            frame = cv2.resize(frame, dsize=(width,height), interpolation=cv2.INTER_AREA)
            input[i] = frame
            cv2.waitKey(1)
    cap.release()
    
    input = torch.from_numpy(process_video(input)).reshape((1,3,64,128,128))
    input = input/255.0 # (64,128,128,3) -> (1,3,64,128,128)
    input = torch.nn.functional.pad(input,(0,0,0,0,0,0,1,0),"constant",0).float() #0,R,G,B   (1,4,64,128,128)
    
    _model = rPPGNet()
    _model.load_state_dict(torch.load('./model_weights.pt'))
    _model.eval()
    _, _, output, _, _, _, _, _, _ = _model(input)
    
    
    fs=30 #FPS
    low_band = 1.0
    high_band = 3.5
    [b_pulse, a_pulse] = butter(2, [low_band / fs * 2, high_band / fs * 2], btype='bandpass')
    
    #fp:손가락 혈압, acp:팔목 혈압
    output = output.detach().numpy()[0] #pytorch tensor -> numpy array
    output_ppg, output_ppg_2, output_spo2, output_spo2_2 = output[0,:], output[1,:], output[2,:], output[3,:] 


    #photoplethysmography(PPG) -> heart rate(HR)
    output_ppg = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(output_ppg))
    #heart_rate,_,_,_,_ = BPM(data=output_ppg, fps=fs, startTime=0, minHz=low_band, maxHz=high_band, verb=False).BVP_to_BPM()
    heart_rate = find_max_magnitude_hz(output_ppg,fs, low_band, high_band)*60
 
    #SpO2 processing
    output_spo2 = (output_spo2-np.min(output_spo2))/(np.max(output_spo2)-np.min(output_spo2)) *10 +90
    output_spo2 = np.median(output_spo2)#np.mean(output_spo2)
      
    print("HR:",heart_rate,"  SPO2:",output_spo2)

if __name__ == '__main__':
    model(sys.argv[1])
