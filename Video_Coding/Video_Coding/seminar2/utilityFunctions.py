import numpy as np
import os


##used
def getYCC(frame):    
    """Converts an RGB image frame into Y Cb and Cr Components

    Parameters
    ----------
    frame : 3 Dimentional Numpy Array
    Returns
    -------
    Y
        1-D Numpy array of type uint8
    Cb
        1-D Numpy array of type int8
    Cr
        1-D Numpy array of type int8
    """
    # Y=np.array(np.zeros(frame[:,:,0].shape),dtype='uint8')
    Y=(np.uint8(0.114*frame[:,:,0])+np.uint8(0.587*frame[:,:,1])+np.uint8(0.299*frame[:,:,2]))/255

    #Cb = 0.564(B-Y)
    Cb=np.array(np.zeros(frame[:,:,0].shape),dtype='int8')
    Cb=(0.564*(frame[:,:,0]/255-Y))
    
    #Cr = (0.713(R-Y))
    Cr=np.array(np.zeros(frame[:,:,2].shape),dtype='int8')
    Cr=(0.713*(frame[:,:,2]/255-Y))
    return Y,Cb,Cr



##used
def convertToRGBFrame(Y,Cb,Cr):
    """Converts Y, Cb and Cr components into an RGB frame

    Parameters
    ----------
    Y
        1-D Numpy array of type uint8
    Cb
        1-D Numpy array of type int8
    Cr
        1-D Numpy array of type int8
    Returns
    -------
    frameRGB
        3-D Numpy array of type
    """
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
##used
def getSize(filename):
    st = os.stat(filename)
    return st.st_size

# put everything together in one method
def chroma_subsampling_4_2_0(Y,C_b,C_r,N):
    """ returns each component with applied 4:2:0 sampling including zeros"""
    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s =  np.zeros((Y.shape[0],Y.shape[1])) # TBD
    C_b_s[::N,::N] =C_b[::N,::N]
    C_r_s =  np.zeros((Y.shape[0],Y.shape[1])) # TBD
    C_r_s[::N,::N] =C_r[::N,::N]

    assert((np.array(C_b_s.shape[0:2]) == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])

# put everything together in one method
def chroma_subsampling_4_2_0_comp(Y,C_b,C_r,N):
    """ returns each component with applied 4:2:0 sampling without zeros"""
    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s =  np.zeros((Y.shape[0]//N,Y.shape[1]//N)) # TBD
    C_b_s[::,::] =C_b[::N,::N]
    C_r_s =  np.zeros((Y.shape[0]//N,Y.shape[1]//N)) # TBD
    C_r_s[::,::] =C_r[::N,::N]

    assert((np.array(C_b_s.shape[0:2]) * N == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) * N == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])

def chroma_upsample_4_2_0(Y,C_b,C_r,N):
    """ returns each component with applied 4:2:0 sampling"""
    
    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s =  np.zeros((Y.shape[0],Y.shape[1])) # TBD
    C_b_s[::N,::N] =C_b[::,::]
    C_r_s =  np.zeros((Y.shape[0],Y.shape[1])) # TBD
    C_r_s[::N,::N] =C_r[::,::]

    assert((np.array(C_b_s.shape[0:2]) == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])