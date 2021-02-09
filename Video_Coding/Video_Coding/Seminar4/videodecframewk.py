import utilityFunctions as utFuncs
import numpy as np
import cv2
import sys
import pickle
import scipy.signal
from utilityFunctions import view_as_block

# Program to open a video input file 'videorecord.txt' (python txt format using pickle) and display it on the screen.
# This is a framework for a simple video decoder to build.
# Gerald Schuller, April 2015
# sampling factor
N = 2
f = open('videorecord_DS_compressed.txt', 'rb')

# Low Pass Kernel:
# rectangular filter kernel:
filt1 = np.ones((2, 2))/2
# Triangular filter kernel
pyramid_filt = scipy.signal.convolve2d(filt1, filt1)/2
filt = pyramid_filt
filteron = True
dctOn = True  # to turn dct on and off


while(True):
    # load next frame from file f and "de-pickle" it, convert from a string back to matrix or tensor:
    # reduced=pickle.load(f)
    Y = pickle.load(f)
    Cb = pickle.load(f)
    Cr = pickle.load(f)

    p = 3  # int he for p/q
    q = 4  # for removing p/q frequencies e.g removing 3/4 = onöy 25% lower frequencies p=3,q=4,,,, for 1/2=50% p=1, q=2

    # #Descaling
    # Y=Y #/255
    # Cb=Cb #/127
    # Cr=Cr #/127
    if dctOn:
        l = 0
        m = 0
        newY = np.zeros((Y.shape[0]*q, Y.shape[1]*q))
        for i, j, b in view_as_block(Y, (8//q, 8//q)):
            new8block = np.zeros((8, 8))
            l = i*q
            m = j*q
            new8block[0:8//q, 0:8//q] = b
            newY[l:l+8, m:m+8] = utFuncs.idct_2d(new8block)

        newCb = np.zeros((Cb.shape[0]*q, Cb.shape[1]*q))
        for i, j, b in view_as_block(Cb, (8//q, 8//q)):
            new8block = np.zeros((8, 8))
            l = i*q
            m = j*q
            new8block[0:8//q, 0:8//q] = b
            newCb[l:l+8, m:m+8] = utFuncs.idct_2d(new8block)

        newCr = np.zeros((Cr.shape[0]*q, Cr.shape[1]*q))
        for i, j, b in view_as_block(Cr, (8//q, 8//q)):
            new8block = np.zeros((8, 8))
            l = i*q
            m = j*q
            new8block[0:8//q, 0:8//q] = b
            newCr[l:l+8, m:m+8] = utFuncs.idct_2d(new8block)

    Y = newY
    Cb = newCb
    Cr = newCr

    # To take inverse dct without using small
   # [Y,Cb,Cr] = utFuncs.inverseDCT(Y,Cb,Cr)

    # cv2.imshow('Cb Component',Cb/255)
    # cv2.imshow('Cr Component',Cr/255)
    [Y_up, Cb_up, Cr_up] = utFuncs.chroma_upsample_4_2_0(Y, Cb, Cr, N)
    if filteron == True:
        Cb_filt = scipy.signal.convolve2d(Cb_up, filt, mode='same')
        Cr_filt = scipy.signal.convolve2d(Cr_up, filt, mode='same')
    else:
        Cb_filt = Cb_up.copy()
        Cr_filt = Cr_up.copy()

    #cv2.imshow('Y Component',Y_up)
    #cv2.imshow('Cb Component',Cb_filt)
    #cv2.imshow('Cr Component',Cr_filt)

    # here goes the decoding:
    framedec = utFuncs.convertToRGBFrame(Y_up, Cb_filt, Cr_filt)
    cv2.imshow('Recovered RGB', framedec/255)
    # Wait for key for 50ms, to get about 20 frames per second playback
    # (depends also on speed of the machine, and recording frame rate, try out):
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
f.close()
cv2.destroyAllWindows()
