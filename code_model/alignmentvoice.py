from os import walk
import librosa
import numpy as np
import os
from audtorch.metrics.functional import pearsonr
import torch
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recordpath="dataset/train/record"
voicepath="dataset/train/voice"
adjustvoicepath="dataset/train/adjustvoice"
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

                    voicefile = "V" + f[prefixlen:]
                    print("voicefile:",voicefile)
                    adjustfile0 = adjustvoicepath + "0/AD"+prefix+ voicefile
                    adjustfile1 = adjustvoicepath + "1/AD" + prefix + voicefile
                    adjustfile2 = adjustvoicepath + "2/AD" + prefix + voicefile
                    adjustfile3 = adjustvoicepath + "3/AD" + prefix + voicefile
                    if os.path.exists(adjustfile0):
                        continue
                    y,rate=librosa.load(recordpath+"/"+f,sr=16000,mono=False)
                    for i in range(0,4):
                        maxy=np.max(y[i])
                        miny=np.min(y[i])
                        if maxy<-miny:
                            maxy=-miny
                        y[i]=y[i]/maxy
                    x, r = librosa.load(voicepath + "/" + voicefile, sr=16000, mono=False)
                    maxx=np.max(x)
                    minx=np.min(x)
                    if maxx<-minx:
                        maxx=-minx
                    x = x / maxx
                    print("max x=",np.max(x), np.min(x))


                    ps=[]
                    recordlen=len(y[0])
                    treate = recordlen - 2 * delay - 1050
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


                            error[delay+i]=my_error[0].data

                        err=np.array(error)
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

                    npps2=np.array([ps[0],ps[1],ps[3]])
                    print("std=",npps.std(),npps,npps1.std(),npps2.std())
                    if not (npps.std()<6):
                        print(f)
                    else:

                        outdata0=np.zeros(recordlen,dtype=np.float32)
                        outdata1 = np.zeros(recordlen, dtype=np.float32)
                        outdata2 = np.zeros(recordlen, dtype=np.float32)
                        outdata3 = np.zeros(recordlen, dtype=np.float32)
                        if ps[0]>0 and ps[1]>0 and ps[2]>0 and ps[3]>0:
                            for j in range(recordlen-ps[0]):
                                if ps[0]+j<len(x):
                                    outdata0[ps[0]+j]=x[j]
                            for j in range(recordlen-ps[1]):
                                if ps[1]+j<len(x):
                                    outdata1[ps[1]+j]=x[j]
                            for j in range(recordlen-ps[2]):
                                if ps[2]+j<len(x):
                                    outdata2[ps[2]+j]=x[j]
                            for j in range(recordlen-ps[3]):
                                if ps[3]+j<len(x):
                                    outdata3[ps[3]+j]=x[j]
                        elif ps[0]<0 and ps[1]<0 and ps[2]<0 and ps[3]<0:
                            for j in range(0,len(x)+ps[0]):
                                if j<recordlen:
                                    outdata0[j]=x[j-ps[0]]
                            for j in range(0,len(x)+ps[1]):
                                if j<recordlen:
                                    outdata1[j]=x[j-ps[1]]
                            for j in range(0,len(x)+ps[2]):
                                if j<recordlen:
                                    outdata2[j]=x[j-ps[2]]
                            for j in range(0,len(x)+ps[3]):
                                if j<recordlen:
                                    outdata3[j]=x[j-ps[3]]
                        adjustvoicedata0=np.asfortranarray(outdata0)
                        adjustvoicedata1 = np.asfortranarray(outdata1)
                        adjustvoicedata2 = np.asfortranarray(outdata2)
                        adjustvoicedata3 = np.asfortranarray(outdata3)
                        datastd=np.std(adjustvoicedata0)
                        print("stddata=",datastd,f)
                        if datastd>0:

                            sf.write(adjustfile0, adjustvoicedata0, samplerate=16000)
                            sf.write(adjustfile1, adjustvoicedata1, samplerate=16000)
                            sf.write(adjustfile2, adjustvoicedata2, samplerate=16000)
                            sf.write(adjustfile3, adjustvoicedata3, samplerate=16000)
                            print("p=",ps[0])

                except Exception as e:
                  print("error",str(e),f)
                k+=1
                print(k)