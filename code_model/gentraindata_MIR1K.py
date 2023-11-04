#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from librosa.util import find_files
from librosa.core import load
import os.path
import utils.myutil as myutil


PATH_iKala = "../dataset/train/mir1k/"

audiolist = find_files(PATH_iKala, ext="wav")
for audiofile in audiolist:
    fname = os.path.split(audiofile)[-1]
    print("Processing: %s" % fname)
    y, _ = load(audiofile, sr=None, mono=False)

    instdrug = np.asfortranarray(y[0])
    instbass=np.asfortranarray(y[0])
    instother=np.asfortranarray(y[0])

    vocal = np.asfortranarray(y[1])

    mix = y[0]+y[1]

    myutil.SaveSpectrogram(mix, vocal, instdrug,instbass,instother, fname)
