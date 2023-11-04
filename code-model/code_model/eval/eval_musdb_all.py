import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
from os import walk
import utils.myutil as myutil
import time

sample_rate = 16000

gsdrsum=np.zeros(2)
gnsdrsum=np.zeros(2)
gsirsum=np.zeros(2)
gsarsum=np.zeros(2)
def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gsdr,gnsdr, gsir, gsar = np.zeros(2),np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    # for i in range(2):
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), True)
    nsdr = sdr - sdr_mixed
    print("sdr", sdr, "sdr_mixed", sdr_mixed)
    gsdr+=len_cropped*sdr
    gnsdr += len_cropped * nsdr
    gsir += len_cropped * sir
    gsar += len_cropped * sar
    total_len += len_cropped
    gsdr=gsdr/total_len
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    print('GSDR:',gsdr)
    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)
    for i in range(0,2):
        gsdrsum[i]+=gsdr[i]
        gnsdrsum[i]+=gnsdr[i]
        gsirsum[i]+=gsir[i]
        gsarsum[i]+=gsar[i]

PATH="../dataset/test/musdb/"
i=0
try:
    for (root,dirs,files) in walk(PATH):

        for f in files:

            if f.endswith(".wav"):
                i+=1
                filename = f
                fname="fname.wav"
                wavfile = PATH+f
                pred_src1_wav1 = '../results/inst-fname.wav'
                pred_src2_wav1 = '../results/vocal-fname.wav'

                mag,magr,magi, phase,cosx,sinx = myutil.LoadAudio(wavfile)

                start = 0
                END = len(mag[0])
                end = END - (END) % 64
                mask, maskr, maski = myutil.ComputeMask(mag[:, start:end], cosx[:, start:end], sinx[:, start:end])


                myutil.SaveAudio("../results/vocal-%s" % fname, mag[:, start:end], mask[:, start:end], phase[:, start:end],
                                 maskr[:, start:end], maski[:, start:end], cosx[:, start:end], sinx[:, start:end],
                                 vocal=True)

                myutil.SaveAudio(
                    "../results/inst-%s" % fname, mag[:, start:end], mask[:, start:end], phase[:, start:end], maskr[:, start:end],
                    maski[:, start:end], cosx[:, start:end], sinx[:, start:end], vocal=False)

                y,_ = librosa.load(wavfile,sr= sample_rate,mono=False)
                src1_wav=y[1]+y[2]+y[3]
                src2_wav=y[4]
                mixed_wav=y[1]+y[2]+y[3]+y[4]
                t1 = time.time()
                maxd = np.max(mixed_wav)
                src1_wav /= maxd
                src2_wav /= maxd
                mixed_wav /= maxd
                pred_src1_wav,_ = librosa.load(pred_src1_wav1,sr= sample_rate,mono=True)
                pred_src2_wav,_ = librosa.load(pred_src2_wav1,sr= sample_rate,mono=True)
                bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav)
                t2 = time.time()
                print("i=",i,f,t2-t1)
                print("gsdrsum=",gsdrsum/i)
                print("gnsdrsum=",gnsdrsum/i)
                print("gsirsum=",gsirsum/i)
                print("gsarsum=",gsarsum/i)
except Exception as e:
    print("error", str(e), f)


