from os import walk
import librosa
import numpy as np
import os
from audtorch.metrics.functional import pearsonr
import torch
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

recordpath="dataset/train/record"
musicpath="dataset/train/music"
adjustmusicpath="dataset/train/adjustmusic"
delay=16000
treate=48000
k=0
#prefix="RMV_front50_M"
#prefix="RMV_front30_M"
#prefix="RMV_front30_left30a_M"
prefix="RMV_front30_right30a_M"

prefixlen=len(prefix)
for (root, dirs, files) in walk(recordpath):
        for f in files:
            print(f[:prefixlen])
            if f.endswith(".wav") and f[:prefixlen]==prefix:

                try:
                    musicfile = "M" + f[prefixlen:]
                    print("musicfile:",musicfile)
                    adjustfile = adjustmusicpath + "/AD"+prefix+ musicfile
                    if os.path.exists(adjustfile):
                        continue
                    y,rate=librosa.load(recordpath+"/"+f,sr=16000,mono=False)
                    x, r = librosa.load(musicpath + "/" + musicfile, sr=16000, mono=False)
                    x = x/2.5
                    print("max x=",np.max(x), np.min(x))

                    ps=[]
                    recordlen=len(y[3])
                    treate = recordlen - 2 * delay - 1050
                    errorlist=[[0.0]*2*delay,[0.0]*2*delay,[0.0]*2*delay,[0.0]*2*delay]
                    error4sum = [0.0] * 2 * delay

                    for m in range(4):
                        error = [0.0]*2*delay

                        record=y[m]


                        for i in range(-delay,delay):
                            rs = record[delay:treate + delay]

                            vs = x[-i + delay:treate - i + delay]
                            rs=torch.tensor(rs)
                            vs=torch.tensor(vs)
                            rs=rs.to(device)
                            vs=vs.to(device)
                            my_error=pearsonr(rs,vs)
                            #error[delay+i]=np.linalg.norm(rs - vs)
                            error[delay+i]=my_error[0].data

                        max=-100
                        p=0
                        for i in range(len(error)):
                            if max<error[i]:
                                max=error[i]
                                p=i
                        print("max=",max,p)
                        ps.append(p-delay)

                    npps=np.array(ps)
                    npps1=np.array(ps[:3])
                    npps2=np.array(ps[1:])
                    npps3=np.array(ps[:2])
                    npps4=np.array((ps[0],ps[1],ps[3]))
                    print("std=",npps.std(),npps)
                    print("ps[2]=",ps[2],"p=",p-delay)
                    
                    if npps.std()>20 and npps1.std()>20 and npps2.std()>20 and npps4.std()>20:
                        print(f)
                    elif abs(p-delay-ps[2])<6 or npps2.std()<6 or npps3.std()<6 or npps4.std()<6:
                        outdata=np.zeros(recordlen,dtype=np.float32)
                        if ps[0]>0 and ps[1]>0 and ps[2]>0 and ps[3]>0:
                            for j in range(recordlen-ps[1]):
                                if ps[1]+j<len(x):
                                    outdata[ps[1]+j]=x[j]
                        elif ps[0]<0:
                            for j in range(0,len(x)+ps[1]):
                                if j<recordlen:
                                    outdata[j]=x[j-ps[1]]
                        adjustmusicdata=np.asfortranarray(outdata)
                        datastd=np.std(adjustmusicdata)
                        print("stddata=",datastd,f)
                        if datastd>0:
                            sf.write(adjustfile, adjustmusicdata, samplerate=16000)
                            print("p=",ps[3])
                except Exception as e:
                  print("error",str(e),f)
                k+=1
                print(k)