import utilityFunctions as utFuncs
import numpy as np
import cv2
import sys
import pickle
import scipy.signal


def decodeAndDisplay(Y,Cb,Cr):
    #Low Pass Kernel:
    #rectangular filter kernel:
    filt1=np.ones((2,2))/2;
    #Triangular filter kernel:
    tri_filt=scipy.signal.convolve2d(filt1,filt1)/2
    filteron=True;

    #load next frame from file f and "de-pickle" it, convert from a string back to matrix or tensor:

    [Y_up,Cb_up,Cr_up]=utFuncs.chroma_upsample_4_2_0(Y,Cb,Cr)
    
    filt=tri_filt;

    if filteron==True:
        Cb_filt=scipy.signal.convolve2d(Cb_up,filt,mode='same')
        Cr_filt=scipy.signal.convolve2d(Cr_up,filt,mode='same')
    else:
        Cb_filt=Cb_up.copy()
        Cr_filt=Cr_up.copy()
    
    #cv2.imshow('Y Component',Y_up)
    #cv2.imshow('Cb Component',Cb_filt)
    #cv2.imshow('Cr Component',Cr_filt)

    # here goes the decoding:
    framedec = utFuncs.convertToRGBFrame(Y_up,Cb_filt,Cr_filt)
    cv2.imshow('Recovered RGB',framedec)
    #Wait for key for 50ms, to get about 20 frames per second playback 
    #(depends also on speed of the machine, and recording frame rate, try out):
    return

    