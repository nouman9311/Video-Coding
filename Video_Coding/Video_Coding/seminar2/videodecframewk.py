import utilityFunctions as utFuncs
import numpy as np
import cv2
import sys
import pickle
import scipy.signal

#Program to open a video input file 'videorecord.txt' (python txt format using pickle) and display it on the screen.
#This is a framework for a simple video decoder to build.
#Gerald Schuller, April 2015
#sampling factor
N=2
f=open('videorecord_DS_compressed.txt', 'rb')

#Low Pass Kernel:
#rectangular filter kernel:
filt1=np.ones((2,2))/2
#Triangular filter kernel
pyramid_filt=scipy.signal.convolve2d(filt1,filt1)/2
filt=pyramid_filt
filteron=True


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
    [Y_up,Cb_up,Cr_up]=utFuncs.chroma_upsample_4_2_0(Y,Cb,Cr,N)
    


    if filteron==True:
       Cb_filt=scipy.signal.convolve2d(Cb_up,filt,mode='same')
       Cr_filt=scipy.signal.convolve2d(Cr_up,filt,mode='same')
    else:
        Cb_filt=Cb_up.copy()
        Cr_filt=Cr_up.copy()


    # cv2.imshow('Y Component',Y_up)
    #cv2.imshow('Cb Component',Cb_filt)
    #cv2.imshow('Cr Component',Cr_filt)

    # here goes the decoding:
    framedec = utFuncs.convertToRGBFrame(Y_up,Cb_filt,Cr_filt)
    cv2.imshow('Recovered RGB',framedec)
    #Wait for key for 50ms, to get about 20 frames per second playback 
    #(depends also on speed of the machine, and recording frame rate, try out):
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
