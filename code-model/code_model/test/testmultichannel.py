#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import code_model.utils.myutil as myutil
f="amy_4_02.wav"
fname = '../dataset/test/multichannel/RMV_front30_left30a_M-'+f

mag,ild, phase,cos2,sin2 = myutil.LoadAudio3D(fname)

END=len(mag[0][0])
start = 0
end = start+END-END%64

mag2, costheta, sintheta= myutil.Compute3DMask(mag[:,:, start:end],ild[:,:,start:end],cos2[1:,start:end],sin2[1:,start:end])

myutil.SaveAudio3D("../results/vocal3d-%s" % f, mag[2,:, start:end]*mag2, phase[:,:, start:end],costheta,sintheta,vocal=True)

myutil.SaveAudio3D(
    "../results/inst3d-%s" % f, mag[2,:, start:end]*(1-mag2), phase[:,:, start:end],costheta,sintheta,vocal=False)



