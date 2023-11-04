import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np

sample_rate = 16000

wavfile = '../dataset/test/musdb/mus-Music-1.wav'
pred_src1_wav1 = '../results/inst-mus-Music-1.wav'
pred_src2_wav1 = '../results/vocal-mus-Music-1.wav'


y,_ = librosa.load(wavfile,sr= sample_rate,mono=False)
src1_wav=y[1]+y[2]+y[3]

src2_wav=y[4]

mixed_wav=y[1]+y[2]+y[3]+y[4]

maxd=np.max(mixed_wav)
src1_wav/=maxd
src2_wav/=maxd
mixed_wav/=maxd
pred_src1_wav,_ = librosa.load(pred_src1_wav1,sr= sample_rate,mono=True)
pred_src2_wav,_ = librosa.load(pred_src2_wav1,sr= sample_rate,mono=True)
start=0
def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    print(len_cropped)
    src1_wav = src1_wav[start:start+len_cropped]
    print(len(src1_wav))
    src2_wav = src2_wav[start:start+len_cropped]
    mixed_wav = mixed_wav[start:start+len_cropped]
    gsdr,gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0

    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), True)
    print("sdr",sdr,"sdr_mixed",sdr_mixed)
    nsdr = sdr - sdr_mixed
    gsdr+=len_cropped*sdr
    gnsdr += len_cropped * nsdr
    gsir += len_cropped * sir
    gsar += len_cropped * sar
    total_len += len_cropped
    gsdr=gsdr/total_len
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    print("GSDR:",gsdr)
    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)


if __name__ == '__main__':
    bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav)
