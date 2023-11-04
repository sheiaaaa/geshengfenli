from os import walk
import librosa
import numpy as np
import os
import struct
import math
recordpath="dataset/train/record"
voicepath="dataset/train/voice"
adjustmusicpath="dataset/train/adjustmusic"

adjustvoicepath0="dataset/train/adjustvoice0"
adjustvoicepath1="dataset/train/adjustvoice1"
adjustvoicepath2="dataset/train/adjustvoice2"
adjustvoicepath3="dataset/train/adjustvoice3"
channel8record="dataset/train/multichannelrecord"
noisefile="dataset/train/record/RMV_front30_left30a_M-abjones_2_08.wav"
#prefix="ADRMV_front50_MV"
#rprefix="RMV_front50_M"
#prefix="ADRMV_front30_MV"
#rprefix="RMV_front30_M"
#prefix="ADRMV_front30_left30a_MV"
#rprefix="RMV_front30_left30a_M"
prefix="ADRMV_front30_right30a_MV"
rprefix="RMV_front30_right30a_M"

prefixL=len(prefix)
prefixlen=len(rprefix)
k=0
n,r_=librosa.load(noisefile,sr=16000,mono=False)
noise=[0]*320000
npart=n[1][-5000:]
for i in range(32):
    for j in range(5000):
        noise[i*5000+j]=npart[j]
for (root, dirs, files) in walk(recordpath):
        for f in files:
            if f.endswith(".wav") and f[:prefixlen] == rprefix:
                try:
                    voicefile = f[prefixlen:]
                    print("v:",voicefile)
                    adjustmusicfile = adjustmusicpath + "/"+prefix[:-1]+"M" + voicefile
                    adjustvoicefile0 = adjustvoicepath0 + "/" + prefix + voicefile
                    adjustvoicefile1 = adjustvoicepath1 + "/" + prefix + voicefile
                    adjustvoicefile2 = adjustvoicepath2 + "/" + prefix + voicefile
                    adjustvoicefile3 = adjustvoicepath3 + "/" + prefix + voicefile

                    if  not os.path.exists(adjustvoicefile0):
                        print("no:",adjustvoicefile0)
                        continue
                    print("f:",recordpath+"/"+f)
                    y,rate=librosa.load(recordpath+"/"+f,sr=16000,mono=False)

                    x, r = librosa.load(adjustmusicfile, sr=16000, mono=False)
                    z0, r_ = librosa.load(adjustvoicefile0, sr=16000, mono=False)
                    z1, r_ = librosa.load(adjustvoicefile1, sr=16000, mono=False)
                    z2, r_ = librosa.load(adjustvoicefile2, sr=16000, mono=False)
                    z3, r_ = librosa.load(adjustvoicefile3, sr=16000, mono=False)

                    sample=len(y[0])
                    datalen = len(y[0])*10*2
                    fmt = [b"RIFF", datalen + 36, b"WAVEfmt ", 16, 1,10, 16000, 320000, 20, 16, b"data", datalen]
                    fmttype = ["4s", "i", "8s", "i", "h", "h", "i", "i", "h", "h", "4s", "i"]
                    with open(channel8record+"/"+rprefix+voicefile, "wb") as wf:
                        for i in range(0, 12):
                            d = struct.pack(fmttype[i], fmt[i])
                            wf.write(d)

                        for i in range(0, sample):

                            data = int(y[0][i]*32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(y[1][i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(y[2][i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(y[3][i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)

                            data = int(z0[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(z1[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(z2[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(z3[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(x[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                            data = int(noise[i] * 32767)
                            d = struct.pack("h", data)
                            wf.write(d)
                    print("p=",f)
                except Exception as e:
                    print(e)