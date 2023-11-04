import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
from os import walk
import utils.myutil as myutil

sample_rate = 16000
gnsdrsum=np.zeros(2)
gsirsum=np.zeros(2)
gsarsum=np.zeros(2)
def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    # for i in range(2):
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), True)
    nsdr = sdr - sdr_mixed
    gnsdr += len_cropped * nsdr
    gsir += len_cropped * sir
    gsar += len_cropped * sar
    total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)
    for i in range(0,2):
        gnsdrsum[i]+=gnsdr[i]
        gsirsum[i]+=gsir[i]
        gsarsum[i]+=gsar[i]

PATH="../dataset/test/multichannel/"
i=0
try:
    for (root,dirs,files) in walk(PATH):

        for f in files:

            if f.endswith(".wav"):
                i+=1
                filename = f
                fname="fname.wav"
                wavfile=PATH+f

                pred_src1_wav1 = '../results/inst3d-fname.wav'
                pred_src2_wav1 = '../results/vocal3d-fname.wav'
                mag,ild, phase,cos2,sin2 = myutil.LoadAudio3D(wavfile)

                END = len(mag[0][0])
                start = 0

                end = start + END-END%64

                mask,cos,sin = myutil.Compute3DMask(mag[:, :, start:end],ild[:,:,start:end],cos2[1:,start:end],sin2[1:,start:end])

                myutil.SaveAudio3D("../results/vocal3d-%s" % fname, mag[2, :, start:end] * mask, phase[:, :, start:end],cos,sin,vocal=True)

                myutil.SaveAudio3D(
                    "../results/inst3d-%s" % fname, mag[2, :, start:end] * (1 - mask), phase[:, :, start:end],cos,sin)


                y,_ = librosa.load(wavfile,sr= sample_rate,mono=False)
                src1_wav=y[8]+y[9]
                src2_wav=y[6]
                mixed_wav=y[2]

                pred_src1_wav,_ = librosa.load(pred_src1_wav1,sr= sample_rate,mono=True)
                pred_src2_wav,_ = librosa.load(pred_src2_wav1,sr= sample_rate,mono=True)
                bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav)
                print("i=",i,f)
                print("gnsdrsum=",gnsdrsum/i)
                print("gsirsum=",gsirsum/i)
                print("gsarsum=",gsarsum/i)
except Exception as e:
    print("error", str(e), f)

