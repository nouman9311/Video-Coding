import utilityFunctions as utFuncs
import numpy as np
import cv2
import sys
import pickle

#Program to open a video input file 'videorecord.txt' (python txt format using pickle) and display it on the screen.
#This is a framework for a simple video decoder to build.
#Gerald Schuller, April 2015

f=open('videorecord.txt', 'rb')


while(True):
#load next frame from file f and "de-pickle" it, convert from a string back to matrix or tensor:
    #reduced=pickle.load(f)
    Y= pickle.load(f)
    Cb= pickle.load(f)
    Cr= pickle.load(f)

    #Descaling
    Y=Y/255
    Cb=Cb/127
    Cr=Cr/127
    # cv2.imshow('Y Component',Y/255)
    # cv2.imshow('Cb Component',Cb/255)
    # cv2.imshow('Cr Component',Cr/255)


    # here goes the decoding:
    framedec = utFuncs.convertToRGBFrame(Y,Cb,Cr)
    cv2.imshow('Video',framedec)
    #Wait for key for 50ms, to get about 20 frames per second playback 
    #(depends also on speed of the machine, and recording frame rate, try out):
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
