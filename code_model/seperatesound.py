from os import walk
import librosa
import librosa.display
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
DATA_PATH = 'dataset/train/mir1k'
SAVE_MUSIC_PATH= 'dataset/train/music'
SAVE_VOICE_PATH='dataset/train/voice'

for (root,dirs,files) in walk(DATA_PATH):

    for f in files:
        if f.endswith(".wav"):

                y,sr=librosa.load(root+"/"+f, sr=None, mono=False, offset=0.0, duration=None)
                Y=np.asfortranarray(y[0])
                X=np.asfortranarray(y[1])
                librosa.output.write_wav(SAVE_MUSIC_PATH+"/"+"M-"+f,Y,sr=16000)
                librosa.output.write_wav(SAVE_VOICE_PATH + "/" + "V-" + f, X, sr=16000)