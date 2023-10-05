Make Training DataSets
1. run seperatesound.py   # stereo (dataset/train/mir1k) -> mono (dataset/train/music:accompaniment;dataset/train/voice:singing voice)
2. run recordsound.py     # The computer plays the singing voice, controls the NAO robot to play music and record (the recording file is stored in /home/nao/recordings)
3. Download the recording file  #copy the recording files from /home/nao/recordings (on robot) to the dataset/train/record directory.
4. run alignmentvoice.py  #generate 4 mono aligned singing voice files ((stored in the directories of dataset/train/adjustvoice0, adjustvoice1, adjustvoice2, and adjustvoice3)
5. run alignmentmusic.py  #generate a mono aligned accompaniment files ((stored in the directories of dataset/train/adjustmusic)
6. run alignment10channel.py   #generate 10 channel wave file (training dataset)