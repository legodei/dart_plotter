# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:00:49 2019

@author: Haojie

Processes an mp4 file with trajectory information in it and outputs
a .wav sound file and also a .csv file with the coordinates. 
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.signal import find_peaks
import cv2
from pydub import AudioSegment

# Input Parameters
audio_thresh = 20000 # Audio Cutoff Threshold for shot recognition (see plot)
file_name = 'rifle_2_acc' # Video file name in mp4 format
video_thresh = 70 # Brightness threshold for dart detection (0-255)
shot_gap = 1.2 # minimum number of seconds between shots

# Variable Generation Based on Inputs
video_file = file_name+'.mp4'

# Extract audio file from video, save it in wav format
audio = AudioSegment.from_file(video_file,'mp4')
audio.export(file_name+'.wav','wav')

# Open the Audio File
sound = wave.open(file_name + '.wav','r') #Insert file name here.

# Extract Raw Audio from Wav File
signal_raw = sound.readframes(-1) # read the audio data into a buffer
audio_rate = sound.getframerate() # find the audio frame rate for timestamping
signal_raw = abs(np.frombuffer(signal_raw, np.int16,)) # signal is the raw audio file
signal_right = signal_raw[::2] # gets every 2nd element starting with 0, right ch.
signal_left = signal_raw[1::2] # gets every 2nd element starting with 1, left ch.
signal = signal_right/2+signal_left/2 # finds the average of the two channels.
sound.close() #closes sound file

# Find Local Maxima corresponding to firing events.
shot_times,_ = find_peaks(signal,height = audio_thresh,distance = int(audio_rate*shot_gap))
shot_times_sec = shot_times/audio_rate # timestamp of shot in seconds

# Plots Identified shot times for verification.
plt.figure(1) #places and 'x' on every peak it thinks is a shot'
plt.title('Shot Times')
plt.plot(np.array(list(range(len(signal))))/audio_rate,signal)
plt.plot(shot_times_sec, signal[shot_times], "x")
plt.show()

cap = cv2.VideoCapture(video_file)
video_rate = cap.get(cv2.CAP_PROP_FPS)

shotnumber = 0 # counter for number of shots
fontsize = 0.4 # font size for monitor display
escape = False # boolean state for aborting

trajectories = [] # Initialize Trajectories

for tstmp in shot_times_sec: # Loop over shots using the time stamps
	trajcoordlist = np.zeros((0,2),np.int32) # empty list for storing coords
	shotnumber+=1 # increment the shot numer by 1
	if escape:
		break # press q to abort
	over = False
	for i in range(120): # Loop over 120 frames for each shot.
		if escape or over:
			break
		# grab the frame that we want, by converting timestamp into video frame
		cap.set(1,int(tstmp*video_rate-20+i)) # constant is for adjusting timing
		ret, frame = cap.read() 
		
		#crop the image that we grabbed to the region of interest
		#adjust this according to your mounting position and field of view.
		crop = frame[400:900,660:1260]
		crop2 = crop.copy()
		gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) # convert to gray scale
		gray[250:500,:]=0; # mask the muzzle of the gun out of the picture
		
		# mask more of the frame as the dart moves, prevents dust particles
		# coming out of the barrel from interfering
		if np.size(trajcoordlist) >3:
			gray[200:500,:]=0;
		elif np.size(trajcoordlist) > 5: 
			gray[150:500,:]=0;
			
		# thresholds the dart (with pixels) based on brightness
		thresh = cv2.threshold(gray, video_thresh, 255, cv2.THRESH_BINARY)[1]
		
		# finds blobs of thresholded pixels that are connected
		(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_NONE)
		# initialize the minimum number of pixels we need to recognize as dart
		# helps filter out single pixel noise and dust illuminated by light
		max_area = 4
		dart = None
		
		# loop over all bright blobs we see, and accepts the largest as the dart
		for c in cnts:
			if cv2.contourArea(c)>max_area:
				max_area=cv2.contourArea(c)
				dart = c
				M = cv2.moments(dart)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
		
		# if we pass 40 frames and we don't see the dart, it has left the field of view
		# abort early to avoid having to loop through empty video frames
		if max_area == 4 and i>40:
			over = True

		# if the maximum area was modified, then we have a coordinate to add
		if max_area != 4:	
			trajcoordlist = np.append(trajcoordlist, np.array([[cx,cy]]),axis=0)
		
		# trajectories holds the history of all the shots except for the current one
		# draw all recorded trails in dark red
		for trail in trajectories:
			trail = trail.reshape((-1,1,2))
			cv2.polylines(crop,[trail],False,(0,0,50),1,lineType=4)
		# if we have more than 3 points in the current shot, then start plotting
		# the current trail live.
		if np.size(trajcoordlist)>3:
			cv2.polylines(crop,[trajcoordlist.reshape((-1,1,2))],False,(0,100,100),2,lineType=4)
			
		# displays the current frame number and shot number for user utility
		cv2.putText(crop, "SHOT: "+str(shotnumber),(10,20),cv2.FONT_HERSHEY_SIMPLEX,fontsize, (255,255,0),1)
		cv2.putText(crop, "FRAME: "+str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,fontsize, (255,255,0),1)	  
		thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		
		# if we found a bright blob we think is a dart, display a circle around it 
		if len(max_area != 4):
			cv2.circle(crop,(cx,cy),15,(100,130,0))
			
		# combine the thresholded image with the live image side by side
		monitor = np.hstack((crop,crop2))
		cv2.imshow('frame',monitor)
		
		# takes key input to abort if q is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			escape = True
	trajectories.append(trajcoordlist)
	
# stop viewing and close video windows
cap.release()
cv2.destroyAllWindows()

# outputs a csv file if we didn't abort halfway and finished processing
if not escape:
	output = np.zeros((0,2),np.int32)
	# stiches all trajectories into one list for output
	for path in trajectories:
			output = np.append(output,path,axis=0)
	# saves csv file with all points that were recorded
	np.savetxt(file_name+'.csv', output, delimiter=",")