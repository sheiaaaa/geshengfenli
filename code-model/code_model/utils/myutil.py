#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import soundfile as sf
from librosa.util import find_files
from librosa.core import stft, load, istft, resample


import numpy as np
import code_model.const as C
import os.path

from code_model.model.model96_3D import InceptionResUNet3D
from code_model.model.model92 import InceptionResUNet2D
import torch
from torch import Tensor
import time

def SaveSpectrogram(y_mix, y_vocal, y_inst, fname, original_sr=16000):
    y_mix = resample(y_mix, original_sr, C.SR)
    y_vocal = resample(y_vocal, original_sr, C.SR)
    y_inst = resample(y_inst, original_sr, C.SR)

    S_mix = np.abs(
        stft(y_mix, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_vocal = np.abs(
        stft(y_vocal, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)

    norm = S_mix.max()
    S_mix /= norm
    S_vocal /= norm
    S_inst /= norm

    np.savez(os.path.join(C.PATH_FFT, fname+".npz"),
             mix=S_mix, vocal=S_vocal, inst=S_inst)

def SaveSpectrogram3D_8channel(y_ch0,y_ch1,y_ch2,y_ch3, y_vocal0,y_vocal1,y_vocal2,y_vocal3,fname, original_sr=16000):


    stft_ch0 = stft(y_ch0, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_ch1 = stft(y_ch1, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_ch2 = stft(y_ch2, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_ch3 = stft(y_ch3, n_fft=C.FFT_SIZE, hop_length=C.H)
    S_ch0 = np.abs(stft_ch0).astype(np.float32)
    S_ch1 = np.abs(stft_ch1).astype(np.float32)
    S_ch2 = np.abs(stft_ch2).astype(np.float32)
    S_ch3 = np.abs(stft_ch3).astype(np.float32)

    S_ch2r = (stft_ch2.real).astype(np.float32)

    y_itd23 = np.divide(stft_ch3,stft_ch2,out=np.zeros_like(stft_ch3),where=stft_ch2!=0)
    y_itd21 = np.divide(stft_ch1,stft_ch2,out=np.zeros_like(stft_ch1),where=stft_ch2!=0)#stft_ch1/stft_ch2
    y_itd20 = np.divide(stft_ch0,stft_ch2,out=np.zeros_like(stft_ch0),where=stft_ch2!=0)#stft_ch0/stft_ch2


    stft_v0=stft(y_vocal0, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_v1 = stft(y_vocal1, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_v2 = stft(y_vocal2, n_fft=C.FFT_SIZE, hop_length=C.H)
    stft_v3 = stft(y_vocal3, n_fft=C.FFT_SIZE, hop_length=C.H)
    v_itd23 = np.divide(stft_v3, stft_v2, out=np.zeros_like(stft_v3), where=stft_v2 != 0)
    v_itd21 = np.divide(stft_v1, stft_v2, out=np.zeros_like(stft_v1), where=stft_v2 != 0)  # stft_ch1/stft_ch2
    v_itd20 = np.divide(stft_v0, stft_v2, out=np.zeros_like(stft_v0), where=stft_v2 != 0)  # stft_ch0/stft_ch2
    S_vocal0 = np.abs(stft_v0).astype(np.float32)
    S_vocal1 = np.abs(stft_v1).astype(np.float32)
    S_vocal2 = np.abs(stft_v2).astype(np.float32)
    S_vocal3 = np.abs(stft_v3).astype(np.float32)
    S_itd23 = np.abs(y_itd23).astype(np.float32)
    S_itd20 = np.abs(y_itd20).astype(np.float32)
    S_itd21 = np.abs(y_itd21).astype(np.float32)


    S_ild20 = np.sin(np.angle(y_itd20)).astype(np.float32)
    S_ild23 = np.sin(np.angle(y_itd23)).astype(np.float32)
    S_ild21 = np.sin(np.angle(y_itd21)).astype(np.float32)
    S_cos20 = np.cos(np.angle(y_itd20)).astype(np.float32)
    S_cos21 = np.cos(np.angle(y_itd21)).astype(np.float32)
    S_cos23 = np.cos(np.angle(y_itd23)).astype(np.float32)



    V_ild0 = np.cos(np.angle(stft_ch2)).astype(np.float32)
    V_ild1 = (stft_v2).real.astype(np.float32)
    V_ild3 = np.sin(np.angle(stft_ch2)).astype(np.float32)
    V_ild4 = (stft_v2).imag.astype(np.float32)
    V_ild5 = np.cos(np.angle(stft_ch3)).astype(np.float32)


    max00=S_vocal0.max()
    max1=S_ch1.max()
    max0=S_ch0.max()
    max2=S_ch2.max()
    max3=S_ch3.max()

    norm = np.max(np.array([max00,max0,max1,max2,max3]))


    S_ch0 /= norm
    S_ch1 /= norm
    S_ch2 /= norm
    S_ch3 /= norm
    S_vocal0 /= norm
    S_vocal1/=norm
    S_vocal2 /= norm
    S_vocal3 /= norm
    V_ild1/=norm
    V_ild4/=norm

    np.savez(os.path.join(C.PATH_FFT3D, fname+".npz"),
             ch0=S_ch0,ch1=S_ch1,ch2=S_ch2,ch3=S_ch3, ch2r=S_ch2r,vocal=S_vocal0,vocal1=S_vocal1,
             vocal2=S_vocal2,vocal3=S_vocal3,itd0=S_itd23,itd1=S_itd21,itd2=S_itd20,itd3=S_ild23,
             itd4=S_ild21,itd5=S_ild20,ild0=V_ild0,ild1=V_ild1,ild3=V_ild3,ild4=V_ild4,ild5=V_ild5,cos20=S_cos20,cos21=S_cos21,cos23=S_cos23)
def LoadDataset(target="vocal"):
    filelist_fft = find_files(C.PATH_FFT, ext="npz")[:1000]

    Xlist = []
    Ylist = []
    i=0
    for file_fft in filelist_fft:

        if len(file_fft)>0:
            dat = np.load(file_fft,mmap_mode="r")

            if len(dat["mix"][0])>C.PATCH_LENGTH-10:

                Xlist.append(dat["mix"])

                if target == "vocal":
                    assert(dat["mix"].shape == dat["vocal"].shape)
                    Ylist.append(dat["vocal"])
                else:
                    assert(dat["mix"].shape == dat["inst"].shape)
                    Ylist.append(dat["inst"])
                i+=1
                if i>=1000:
                    break
    print("len x=",len(Xlist))

    return Xlist, Ylist

def LoadDataset3D_8channel(target="vocal"):
    index=0
    filelist_fft = find_files(C.PATH_FFT3D, ext="npz")[index:index+448]

    Xlistch0 = []
    Xlistch1 = []
    Xlistch2 = []
    Xlistch3 = []
    Xlistch2r=[]

    Ylist0 = []
    Ylist1 = []
    Ylist2 = []
    Ylist3 = []
    Xitd0=[]
    Xitd1=[]
    Xitd2=[]
    Xitd3=[]
    Xitd4=[]
    Xitd5=[]
    Ild0=[]
    Ild1=[]
    Ild3=[]
    Ild4=[]
    Ild5=[]
    cos20=[]
    cos21=[]
    cos23=[]

    i=0
    for j in range(0,448):
        k=random.randint(0,446)
        file_fft=filelist_fft[k]

        dat = np.load(file_fft)

        if len(dat["ch0"][0])>C.PATCH_LENGTH-10:

            Xlistch0.append(dat["ch0"])
            Xlistch1.append(dat["ch1"])
            Xlistch2.append(dat["ch2"])
            Xlistch3.append(dat["ch3"])
            Xlistch2r.append(dat["ch2r"])

            Xitd0.append(dat["itd0"])
            Xitd1.append(dat["itd1"])
            Xitd2.append(dat["itd2"])
            Xitd3.append(dat["itd3"])
            Xitd4.append(dat["itd4"])
            Xitd5.append(dat["itd5"])
            Ild0.append(dat["ild0"])
            Ild1.append(dat["ild1"])
            Ild3.append(dat["ild3"])
            Ild4.append(dat["ild4"])
            Ild5.append(dat["ild5"])
            cos20.append(dat["cos20"])
            cos21.append(dat["cos21"])
            cos23.append(dat["cos23"])


            if target == "vocal":
                assert(dat["ch1"].shape == dat["vocal"].shape)
                Ylist0.append(dat["vocal"])
                Ylist1.append(dat["vocal1"])
                Ylist2.append(dat["vocal2"])
                Ylist3.append(dat["vocal3"])

            i+=1
            if i>=448:
                break
    print("len x=",len(Xlistch0))

    return Xlistch0,Xlistch1,Xlistch2,Xlistch3,Xlistch2r,Ylist0,Ylist1,Ylist2,Ylist3,Xitd0,Xitd1,Xitd2,Xitd3, \
           Xitd4,Xitd5,Ild0,Ild1,Ild3,Ild4,Ild5,cos20,cos21,cos23
def LoadAudio(fname):

        y, sr = load(fname, sr=C.SR, mono=False)
        spec = stft(np.array(y[0] + y[1]), n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)

        mag = np.abs(spec)
        magr = spec.real
        magi = spec.imag
        cosx = (np.cos(np.angle(spec)))
        sinx = (np.sin(np.angle(spec)))

        phase = np.exp(1.j * np.angle(spec))

        return mag, magr, magi, phase, cosx, sinx

def LoadAudio3D(fname):
    y, sr = load(fname, sr=C.SR,mono=False)

    spec0 = stft(np.array(y[0]), n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    spec1 = stft(np.array(y[1]), n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    spec2 = stft(np.array(y[2]), n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    spec3 = stft(np.array(y[3]), n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)

    mag0 = np.abs(spec0)
    mag1 = np.abs(spec1)
    mag2 = np.abs(spec2)
    mag3 = np.abs(spec3)
    cos2=np.cos(np.angle(spec2)).astype(np.float32)
    sin2 = np.sin(np.angle(spec2)).astype(np.float32)
    mag=np.array([mag0,mag1,mag2,mag3])

    max = np.max(mag)
    mag /= max

    y_itd03 = np.divide(spec0, spec3, out=np.zeros_like(spec0), where=spec3 != 0)  # stft_ch0/stft_ch3
    y_itd23 = np.divide(spec3,spec2,out=np.zeros_like(spec3),where=spec2!=0)
    y_itd21 = np.divide(spec1,spec2,out=np.zeros_like(spec1),where=spec2!=0)#stft_ch1/stft_ch2
    y_itd20 = np.divide(spec0,spec2,out=np.zeros_like(spec0),where=spec2!=0)#stft_ch0/stft_ch2

    S_itd23 = (np.abs(y_itd23)).astype(np.float32)
    S_itd20 = (np.abs(y_itd20)).astype(np.float32)
    S_itd21 = (np.abs(y_itd21)).astype(np.float32)


    S_ild20 = np.sin(np.angle(y_itd20)).astype(np.float32)
    S_ild23 = np.sin(np.angle(y_itd23)).astype(np.float32)
    S_ild21 = np.sin(np.angle(y_itd21)).astype(np.float32)
    S_cos20 = np.cos(np.angle(y_itd20)).astype(np.float32)
    S_cos21 = np.cos(np.angle(y_itd21)).astype(np.float32)
    S_cos23 = np.cos(np.angle(y_itd23)).astype(np.float32)


    ild = np.array([S_ild23,S_ild21,S_ild20,S_cos23, S_cos21, S_cos20,cos2,sin2])

    phase0 = np.exp(1.j*np.angle(spec0))

    phase1 = np.exp(1.j * np.angle(spec1))
    phase2 = np.exp(1.j * np.angle(spec2))
    phase3 = np.exp(1.j * np.angle(spec3))
    phase = np.array([phase0, phase1, phase2, phase3])

    return mag,ild, phase,cos2,sin2


def SaveAudio(fname, mag,mask, phase,magr,magi,cosx,sinx,vocal=True):
    if vocal==True:

        mag=mag*mask
        m=(magr*mag*cosx-magi*mag*sinx)+1.j*(magr*mag*sinx+magi*mag*cosx)


    else:

        m=(mag*cosx-magr*mag*mask*cosx+magi*mag*mask*sinx)+1.j*(mag*sinx-magr*mag*mask*sinx-magi*mag*mask*cosx)
    y = istft(m, hop_length=C.H, win_length=C.FFT_SIZE)
    max=np.max(y)
    min=np.min(y)
    if np.abs(min)>max:
        max=np.abs(min)
    y=y/max
    sf.write(fname, y, C.SR)




def Compute3DMask(input_mag,input_ild,cosx,sinx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = InceptionResUNet3D()
    net.load_state_dict(torch.load('../model/model_96_3d.pth',map_location=device))

    net.to(device)
    t0=time.time()
    X=input_mag[np.newaxis, np.newaxis,:, 1:, :]
    Z=input_ild[np.newaxis,np.newaxis,:,1:,:]

    if True:
        X = torch.Tensor(X)

        X = X.to(device)

        Z = torch.Tensor(Z)

        Z = Z.to(device)
        #####################
        cosx=torch.Tensor(cosx)
        cosx=cosx.to(device)
        sinx=torch.Tensor(sinx)
        sinx=sinx.to(device)

        with torch.no_grad():
            mag,mask_cos2,mask_sin2= net(X, Z)


    costheta = mask_cos2*cosx-mask_sin2*sinx
    sintheta = mask_sin2*cosx+mask_cos2*sinx

    mag = mag.data[0, 0, :, :].cpu().numpy()
    mag = np.vstack((np.zeros(mag.shape[1], dtype="float32"), mag))
    costheta = costheta.data[0, 0, :, :].cpu().numpy()
    costheta = np.vstack((np.zeros(costheta.shape[1], dtype="float32"), costheta))
    sintheta = sintheta.data[0, 0, :, :].cpu().numpy()
    sintheta = np.vstack((np.zeros(sintheta.shape[1], dtype="float32"), sintheta))

    t1=time.time()
    print("time=",t1-t0)
    return mag, costheta, sintheta
def ComputeMask(input_mag,magr,magi,voicemodel=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = InceptionResUNet2D()


    #net.load_state_dict(torch.load('../model/model_92_mir1k.pth',map_location=torch.device('cpu')))
    net.load_state_dict(torch.load('../model/model_92_musdb.pth', map_location=torch.device('cpu')))

    net=net.to(device=device)


    X=input_mag[np.newaxis, np.newaxis, 1:, :]

    X=torch.Tensor(X)

    X=X.to(device=device)
    Z = magr[np.newaxis, np.newaxis, 1:, :]
    S= magi[np.newaxis, np.newaxis, 1:, :]

    Z = torch.Tensor(Z)
    S = torch.Tensor(S)
    Z = Z.to(device=device)
    S = S.to(device=device)

    with torch.no_grad():

        mask0,mask1,mask = net(X,Z,S)

        mask = mask.data[0, 0, :, :]
        maskr = (mask0).data[0, 0, :, :]
        maski = (mask1).data[0, 0, :, :]

    mask=mask.cpu().numpy()
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    maskr = maskr.cpu().numpy()
    maskr = np.vstack((np.zeros(maskr.shape[1], dtype="float32"), maskr))
    maski = maski.cpu().numpy()
    maski = np.vstack((np.zeros(maski.shape[1], dtype="float32"), maski))


    return mask,maskr,maski
def SaveAudio3D(fname, mag, phase,cos,sin,vocal=False):

    if vocal==True:
        y2=mag*cos+1.j*(mag*sin)


    else:
        y2 = mag * phase[2]

    y=istft(y2, hop_length=C.H, win_length=C.FFT_SIZE)

    max = np.max(y)
    min = np.min(y)
    if np.abs(min) > max:
        max = np.abs(min)
    y = y / max
    
    sf.write(fname, y, C.SR)

