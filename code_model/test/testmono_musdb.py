#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils.myutil as myutil
fname ="mus-Music-1.wav"
f="../dataset/test/musdb/"+fname

mag,magr,magi, phase,cosx,sinx = myutil.LoadAudio(f)
start = 0
END=len(mag[0])
end = END-(END)%64
mask,maskr,maski = myutil.ComputeMask(mag[:, start:end],cosx[:,start:end],sinx[:,start:end])
myutil.SaveAudio("../results/vocal-%s" % fname, mag[:, start:end],mask[:, start:end], phase[:, start:end],maskr[:,start:end],maski[:,start:end],cosx[:,start:end],sinx[:,start:end],vocal=True)
myutil.SaveAudio(
    "../results/inst-%s" % fname, mag[:, start:end],mask[:, start:end], phase[:, start:end],maskr[:,start:end],maski[:,start:end],cosx[:,start:end],sinx[:,start:end],vocal=False)

