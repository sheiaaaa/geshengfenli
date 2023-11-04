from os import walk
from naoqi import ALProxy     #Nao Robot python lib
import time
import winsound
import os.path
COMPUTER_PATH= "dataset/train/music"
VOICE_PATH="dataset/train/voice"
ROBOT_PATH='/home/nao/recordings'       #Nao Robot directory
try:

    aup=ALProxy("ALAudioPlayer","169.254.179.22",9559)      #Nao Robot IP,port
    record=ALProxy("ALAudioRecorder","169.254.179.22",9559)
except:
    print("ERROR")

path="/home/nao/recordings"
i=0
try:
    for (root,dirs,files) in walk(COMPUTER_PATH):

        for f in files:

            if f.endswith(".wav"):

                filename = ROBOT_PATH + "/" + f

                voicefilename=f
                voicefilename=VOICE_PATH+"/"+voicefilename.replace("M","V")

                filesize=os.path.getsize(voicefilename)
                filetime=(filesize-36)/64000.0

                savefilename = path + "/RMV_front50/RM50" + f

                fileId = aup.loadFile(filename)
                filelength = aup.getFileLength(fileId)
                time.sleep(0.3)
                winsound.PlaySound(voicefilename, winsound.SND_ASYNC)
                aup.post.play(fileId, 1.0, -1.0)

                delay = 0.08
                time.sleep(delay)

                record.post.startMicrophonesRecording(savefilename, 'wav', 16000, (1, 1, 1, 1))

                time.sleep(filetime)
                record.stopMicrophonesRecording()
                time.sleep(1)
                i=i+1
                print(i,filename)

except Exception as e:
    print("error",str(e))



