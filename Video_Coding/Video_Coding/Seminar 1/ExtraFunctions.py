import numpy as np
import os


def verticleFlip(frame):
    return frame[::-1,:,:]

def horizontalFlip(frame):
    return frame[:,::-1,:]


def getRed(frame):
    """ This function gives u
    red component of the image """
    return frame[:,:,2]

def getGreen(frame):
    return frame[:,:,1]

def getBlue(frame):
    return frame[:,:,0]

def getYUV(frame):
    Y=(0.114*frame[:,:,0]+0.587*frame[:,:,1]+0.299*frame[:,:,2])/255;        
    #U=B-Y:
    U=frame[:,:,0]/255.0-Y;
    #V=R-Y:
    V=frame[:,:,2]/255.0-Y;
    return Y,U,V

def getYCC(frame):
    # Y=np.array(np.zeros(frame[:,:,0].shape),dtype='uint8')
    Y=(np.uint8(0.114*frame[:,:,0])+np.uint8(0.587*frame[:,:,1])+np.uint8(0.299*frame[:,:,2]))/255

    #Cb = 0.564(B-Y)
    Cb=np.array(np.zeros(frame[:,:,0].shape),dtype='int8')
    Cb=(0.564*(frame[:,:,0]/255-Y))
    
    #Cr = (0.713(R-Y))
    Cr=np.array(np.zeros(frame[:,:,2].shape),dtype='int8')
    Cr=(0.713*(frame[:,:,2]/255-Y))
    return Y,Cb,Cr

# def convertFrameToYCC(frame):
#     frameYCC=np.zeros(frame.shape)
#     frameYCC=np.array(np.zeros(frame.shape),dtype='uint8')
#     Y=(0.114*frame[:,:,0]+0.587*frame[:,:,1]+0.299*frame[:,:,2])        
#     #Cb = 0.564(B-Y)
#     Cb=(0.564(frame[:,:,0]-1)
#     #Cr = (0.713(R-Y))
#     Cr=(0.713(frame[:,:,2]-1)
#     #Cr = Cb
#     frameYCC[:,:,0]=Y
#     frameYCC[:,:,1]=Cb
#     frameYCC[:,:,2]=Cr
   
#     return frameYCC

def convertFrameToRGB(frame):
    frameRGB=np.zeros(frame.shape)
    Y=frame[:,:,0]
    Cb=frame[:,:,1]
    Cr=frame[:,:,2]
    #B = (Cb + Y)/0.564
    B=(Cb+Y)/0.564
    #R = (Cr + Y)/0.713
    R=Cr+Y
    #G = (Y - 0.114B - 0.299R)/0.587
    G=(Y-0.114*B-0.299*R)/0.587

    frameRGB[:,:,0]=B
    frameRGB[:,:,1]=G
    frameRGB[:,:,2]=R
    return frameRGB

def convertToRGBFrame(Y,Cb,Cr):
    frameRGB=np.zeros((480,640,3))
    # #B = (Cb/0.564 + Y)
    B=(Y+(1.772*((Cb))))
    # #R = (Cr/0.713 + Y)
    R=(Y + (1.402*(Cr)))
    # #G = (Y - 0.114B - 0.299R)/0.587
    G=(Y-(0.34414*(Cb))-(0.71414*(Cr)))

    # R=np.int8(Y + 1.4025*Cr)
    # G=np.int8(Y -0.34434*Cb-0.7144*Cr)
    # B=np.int8(Y + 1.7731*Cb)

    frameRGB[:,:,0]=B
    frameRGB[:,:,1]=G
    frameRGB[:,:,2]=R
    return frameRGB

def getSize(filename):
    st = os.stat(filename)
    return st.st_size