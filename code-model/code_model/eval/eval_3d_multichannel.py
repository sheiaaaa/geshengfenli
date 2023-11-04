import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
f="amy_4_02.wav"
sample_rate = 16000
wavfile = 'dataset/train/multichannelrecord/RMV_front30_left30a_M-'+f
pred_src1_wav1 = 'results/inst3d-'+f
pred_src2_wav1 = 'results/vocal3d-'+f

pred_src1_wav,_ = librosa.load(pred_src1_wav1,sr= sample_rate,mono=True)
pred_src2_wav,_ = librosa.load(pred_src2_wav1,sr= sample_rate,mono=True)
y,_ = librosa.load(wavfile,sr= sample_rate,mono=False)
src1_wav=y[8]+y[9]
src2_wav=y[6]
mixed_wav=y[2]


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src2_wav.shape[-1]
    len_cropped=len_cropped-len_cropped%64
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0

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



if __name__ == '__main__':

    bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav)

