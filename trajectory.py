# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:00:49 2019

@author: Haojie
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.signal import find_peaks
import cv2

#Filter Parameters
audio_thresh = 29000


# Process the Audio File
spf = wave.open('smoothbore.wav','r') #Insert file name here.

# Extract Raw Audio from Wav File
signalraw = spf.readframes(-1) # read the audio data into a buffer
audiorate = spf.getframerate() # find the audio frame rate for timestamping
signalraw = abs(np.frombuffer(signalraw, np.int16,)) # signal is the raw audio file
signal1 = signalraw[::2]
signal2 = signalraw[1::2]
signal = signal1/2+signal2/2

#Find Local Maxima corresponding to firing events.
shot_times,_ = find_peaks(signal,height = 22000,distance = int(audiorate*1.7))
shot_times_sec = shot_times/audiorate

plt.figure(1)
plt.title('Shot Times')
plt.plot(np.array(list(range(len(signal))))/audiorate,signal)
plt.plot(shot_times_sec, signal[shot_times], "x")
plt.show()

cap = cv2.VideoCapture('smoothbore.mp4')
shotnumber = 0
fontsize = 0.4
getout = True
threshval = 50

trajectories = []

for tstmp in shot_times_sec:
	trajcoordlist = np.zeros((1,2),np.int32)
	shotnumber+=1
	if not getout:
		break
	for i in range(120):
		if not getout:
			break
		cap.set(1,int(tstmp*240-20+i))
		ret, frame = cap.read()
		crop = frame[400:900,660:1260]
		crop2 = crop.copy()
		gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
		gray[250:500,:]=0;
		thresh = cv2.threshold(gray, threshval, 255, cv2.THRESH_BINARY)[1]
		(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_NONE)
		max_area = 1
		dart = None
		for c in cnts:
			if cv2.contourArea(c)>max_area:
				max_area=cv2.contourArea(c)
				dart = c
				M = cv2.moments(dart)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				if np.size(trajcoordlist)<3:
					trajcoordlist = np.array([[cx,cy]])
					trajcoordlist = np.append(trajcoordlist, np.array([[cx,cy]]),axis=0)
				else:
					trajcoordlist = np.append(trajcoordlist, np.array([[cx,cy]]),axis=0)
		
			

		for trail in trajectories:
			trail = trail.reshape((-1,1,2))
			cv2.polylines(crop,[trail],False,(0,0,50),1,lineType=4)
		if np.size(trajcoordlist)>3:
			cv2.polylines(crop,[trajcoordlist.reshape((-1,1,2))],False,(0,80,80),1,lineType=4)
		cv2.putText(crop, "SHOT # "+str(shotnumber),(10,20),cv2.FONT_HERSHEY_SIMPLEX,fontsize, (255,255,0),1)
		thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		if len(cnts):
			cv2.circle(crop,(cx,cy),15,(100,130,0))
			
		
		monitor = np.hstack((crop,crop2))

		cv2.imshow('frame',monitor)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			getout = False
	trajectories.append(trajcoordlist)
cap.release()
cv2.destroyAllWindows()

final3 = trajectories[0]
for path in trajectories:
	if np.array_equal(final3, path):
		pass
	else:
		final3 = np.append(final3,path,axis=0)
final3 = final3[:-1,:]
