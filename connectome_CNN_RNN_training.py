# Connectome-based CNN-RNN
# 2021.03.16    Angel Canelo

###### import ######################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import callbacks as cb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.layers import TimeDistributed as T
from time import time
from pymatreader import read_mat
####################################
### Functions and Initializations ##
tf.config.experimental.list_physical_devices('GPU')
# LAMINA INITIALIZATION FILTERS
def init_weights_L1R(shape, dtype=None):
    scale = 1/75
    L1R = scale * np.array([[0, 0, 0], [0, -35, 0], [0, 0, 0]]); L1R = L1R[..., np.newaxis, np.newaxis]
    L1R = tf.convert_to_tensor(L1R, dtype='float32')
    return L1R
def init_weights_L2R(shape, dtype=None):
    scale = 1/75
    L2R = scale*np.array([[0,0, 0],[0, -45, 0],[0, 0, 0]]); L2R = L2R[..., np.newaxis, np.newaxis]
    L2R = tf.convert_to_tensor(L2R, dtype='float32')
    return L2R
def init_weights_L3R(shape, dtype=None):
    scale = 1/75
    L3R = scale*np.array([[0,0, 0],[0, -10, 0],[0, 0, 0]]); L3R = L3R[..., np.newaxis, np.newaxis]
    L3R = tf.convert_to_tensor(L3R, dtype='float32')
    return L3R
def init_weights_L5L1(shape, dtype=None):
    scale = 1/75
    L5L1 = scale*np.array([[0,0, 0],[0, 120, 0],[0, 0, 0]]); L5L1 = L5L1[..., np.newaxis, np.newaxis]
    L5L1 = tf.convert_to_tensor(L5L1, dtype='float32')
    return L5L1
def init_weights_L5L2(shape, dtype=None):
    scale = 1/75
    L5L2 = scale*np.array([[0,0, 0],[0, 60, 0],[0, 0, 0]]); L5L2 = L5L2[..., np.newaxis, np.newaxis]
    L5L2 = tf.convert_to_tensor(L5L2, dtype='float32')
    return L5L2
# Outer MEDULLA INITIALIZATION FILTERS
def init_weights_Mi1L1(shape, dtype=None):
    scale = 1/75
    Mi1L1 = scale*np.array([[0,0, 0],[0, 140, 0],[0, 0, 0]]); Mi1L1 = Mi1L1[..., np.newaxis, np.newaxis]
    Mi1L1 = tf.convert_to_tensor(Mi1L1, dtype='float32')
    return Mi1L1
def init_weights_Mi1L5(shape, dtype=None):
    scale = 1/75
    Mi1L5 = scale*np.array([[0,0, 0],[0, 50, 0],[0, 0, 0]]); Mi1L5 = Mi1L5[..., np.newaxis, np.newaxis]
    Mi1L5 = tf.convert_to_tensor(Mi1L5, dtype='float32')
    return Mi1L5
def init_weights_Tm1L2(shape, dtype=None):
    scale = 1/75
    Tm1L2 = scale*np.array([[0,0, 0],[0, 180, 0],[0, 0, 0]]); Tm1L2 = Tm1L2[..., np.newaxis, np.newaxis]
    Tm1L2 = tf.convert_to_tensor(Tm1L2, dtype='float32')
    return Tm1L2
def init_weights_Tm2L2(shape, dtype=None):
    scale = 1/75
    Tm2L2 = scale*np.array([[0,0, 0],[0, 160, 0],[0, 0, 0]]); Tm2L2 = Tm2L2[..., np.newaxis, np.newaxis]
    Tm2L2 = tf.convert_to_tensor(Tm2L2, dtype='float32')
    return Tm2L2
def init_weights_Tm3L1(shape, dtype=None):
    scale = 1/75
    Tm3L1 = scale*np.array([[50,50, 50],[50, 110, 50],[50, 50, 50]]); Tm3L1 = Tm3L1[..., np.newaxis, np.newaxis]
    Tm3L1 = tf.convert_to_tensor(Tm3L1, dtype='float32')
    return Tm3L1
def init_weights_Tm3L5(shape, dtype=None):
    scale = 1/75
    Tm3L5 = scale*np.array([[0,0, 0],[0, 35, 0],[0, 0, 0]]); Tm3L5 = Tm3L5[..., np.newaxis, np.newaxis]
    Tm3L5 = tf.convert_to_tensor(Tm3L5, dtype='float32')
    return Tm3L5
def init_weights_Tm4L2(shape, dtype=None):
    scale = 1/75
    Tm4L2 = scale*np.array([[0,0, 0],[0, 70, 0],[0, 0, 0]]); Tm4L2 = Tm4L2[..., np.newaxis, np.newaxis]
    Tm4L2 = tf.convert_to_tensor(Tm4L2, dtype='float32')
    return Tm4L2
def init_weights_Tm9L3(shape, dtype=None):
    scale = 1/75
    Tm9L3 = scale*np.array([[0,0, 0],[0, 26, 0],[0, 0, 0]]); Tm9L3 = Tm9L3[..., np.newaxis, np.newaxis]
    Tm9L3 = tf.convert_to_tensor(Tm9L3, dtype='float32')
    return Tm9L3
def init_weights_Tm9Mi4(shape, dtype=None):
    scale = 1/75
    Tm9Mi4 = scale*np.array([[0,0, 0],[0, -12, 0],[0, 0, 0]]); Tm9Mi4 = Tm9Mi4[..., np.newaxis, np.newaxis]
    Tm9Mi4 = tf.convert_to_tensor(Tm9Mi4, dtype='float32')
    return Tm9Mi4
def init_weights_Mi9L3(shape, dtype=None):
    scale = 1/75
    Mi9L3 = scale*np.array([[0,0, 0],[0, 60, 0],[0, 0, 0]]); Mi9L3 = Mi9L3[..., np.newaxis, np.newaxis]
    Mi9L3 = tf.convert_to_tensor(Mi9L3, dtype='float32')
    return Mi9L3
def init_weights_Mi4L5(shape, dtype=None):
    scale = 1/75
    Mi4L5 = scale*np.array([[5,5, 5],[5, 20, 5],[5, 5, 5]]); Mi4L5 = Mi4L5[..., np.newaxis, np.newaxis]
    Mi4L5 = tf.convert_to_tensor(Mi4L5, dtype='float32')
    return Mi4L5
def init_weights_C3L1(shape, dtype=None):
    scale = 1/75
    C3L1 = scale*np.array([[0,0, 0],[0, 80, 0],[0, 0, 0]]); C3L1 = C3L1[..., np.newaxis, np.newaxis]
    C3L1 = tf.convert_to_tensor(C3L1, dtype='float32')
    return C3L1
# Inner MEDULLA INITIALIZATION FILTERS
# T4a
def init_weights_T4aMi1(shape, dtype=None):
    scale = 1/75
    T4aMi1 = scale*np.array([[8,8, 0],[8, 32, 0],[8, 24, 0]]); T4aMi1 = T4aMi1[..., np.newaxis, np.newaxis]
    T4aMi1 = tf.convert_to_tensor(T4aMi1, dtype='float32')
    return T4aMi1
def init_weights_T4aTm3(shape, dtype=None):
    scale = 1/75
    T4aTm3 = scale*np.array([[8,0, 8],[0, 10, 0],[0, 0, 0]]); T4aTm3 = T4aTm3[..., np.newaxis, np.newaxis]
    T4aTm3 = tf.convert_to_tensor(T4aTm3, dtype='float32')
    return T4aTm3
def init_weights_T4aMi9(shape, dtype=None):
    scale = 1/75
    T4aMi9 = scale*np.array([[0,0,0,0,0],[0,0,0,-8,0],[0,0,0,-8,-4], [0,0,0,-6,0], [0,0,0,0,0]])
    T4aMi9 = T4aMi9[..., np.newaxis, np.newaxis]
    T4aMi9 = tf.convert_to_tensor(T4aMi9, dtype='float32')
    return T4aMi9
def init_weights_T4aMi4(shape, dtype=None):
    scale = 1/75
    T4aMi4 = scale*np.array([[-4,0, 0],[-6, 0, 0],[-8, 0, 0]]); T4aMi4 = T4aMi4[..., np.newaxis, np.newaxis]
    T4aMi4 = tf.convert_to_tensor(T4aMi4, dtype='float32')
    return T4aMi4
def init_weights_T4aC3(shape, dtype=None):
    scale = 1/75
    T4aC3 = scale*np.array([[-6,0, 0],[-6, 0, 0],[-6, 0, 0]]); T4aC3 = T4aC3[..., np.newaxis, np.newaxis]
    T4aC3 = tf.convert_to_tensor(T4aC3, dtype='float32')
    return T4aC3
# T4b
def init_weights_T4bMi1(shape, dtype=None):
    scale = 1/75
    T4bMi1 = scale*np.array([[0,8, 8],[0, 32, 8],[0, 8, 8]]); T4bMi1 = T4bMi1[..., np.newaxis, np.newaxis]
    T4bMi1 = tf.convert_to_tensor(T4bMi1, dtype='float32')
    return T4bMi1
def init_weights_T4bTm3(shape, dtype=None):
    scale = 1/75
    T4bTm3 = scale*np.array([[0,0, 0],[0, 10, 0],[8, 0, 8]]); T4bTm3 = T4bTm3[..., np.newaxis, np.newaxis]
    T4bTm3 = tf.convert_to_tensor(T4bTm3, dtype='float32')
    return T4bTm3
def init_weights_T4bMi9(shape, dtype=None):
    scale = 1/75
    T4bMi9 = scale*np.array([[0,0,0,0,0],[0,-16,0,0,0],[-8,-16,0,0,0], [0,-16,0,0,0], [0,0,0,0,0]])
    T4bMi9 = T4bMi9[..., np.newaxis, np.newaxis]
    T4bMi9 = tf.convert_to_tensor(T4bMi9, dtype='float32')
    return T4bMi9
def init_weights_T4bMi4(shape, dtype=None):
    scale = 1/75
    T4bMi4 = scale*np.array([[0,0, -8],[0, 0, -8],[0, 0, -8]]); T4bMi4 = T4bMi4[..., np.newaxis, np.newaxis]
    T4bMi4 = tf.convert_to_tensor(T4bMi4, dtype='float32')
    return T4bMi4
def init_weights_T4bC3(shape, dtype=None):
    scale = 1/75
    T4bC3 = scale*np.array([[0,0, -6],[0, 0, -6],[0, 0, -6]]); T4bC3 = T4bC3[..., np.newaxis, np.newaxis]
    T4bC3 = tf.convert_to_tensor(T4bC3, dtype='float32')
    return T4bC3
# T4c
def init_weights_T4cMi1(shape, dtype=None):
    scale = 1/75
    T4cMi1 = scale*np.array([[10,8, 16],[8, 32, 0],[6, 0, 0]]); T4cMi1 = T4cMi1[..., np.newaxis, np.newaxis]
    T4cMi1 = tf.convert_to_tensor(T4cMi1, dtype='float32')
    return T4cMi1
def init_weights_T4cTm3(shape, dtype=None):
    scale = 1/75
    T4cTm3 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]]); T4cTm3 = T4cTm3[..., np.newaxis, np.newaxis]
    T4cTm3 = tf.convert_to_tensor(T4cTm3, dtype='float32')
    return T4cTm3
def init_weights_T4cMi9(shape, dtype=None):
    scale = 1/75
    T4cMi9 = scale*np.array([[0,0, 0],[0, -6, 0],[-8, -6, 0]])
    T4cMi9 = T4cMi9[..., np.newaxis, np.newaxis]
    T4cMi9 = tf.convert_to_tensor(T4cMi9, dtype='float32')
    return T4cMi9
def init_weights_T4cMi4(shape, dtype=None):
    scale = 1/75
    T4cMi4 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]]); T4cMi4 = T4cMi4[..., np.newaxis, np.newaxis]
    T4cMi4 = tf.convert_to_tensor(T4cMi4, dtype='float32')
    return T4cMi4
def init_weights_T4cC3(shape, dtype=None):
    scale = 1/75
    T4cC3 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]]); T4cC3 = T4cC3[..., np.newaxis, np.newaxis]
    T4cC3 = tf.convert_to_tensor(T4cC3, dtype='float32')
    return T4cC3
# T4d
def init_weights_T4dMi1(shape, dtype=None):
    scale = 1/75
    T4dMi1 = scale*np.array([[8,0, 0],[8, 32, 0],[8, 8, 10]]); T4dMi1 = T4dMi1[..., np.newaxis, np.newaxis]
    T4dMi1 = tf.convert_to_tensor(T4dMi1, dtype='float32')
    return T4dMi1
def init_weights_T4dTm3(shape, dtype=None):
    scale = 1/75
    T4dTm3 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]]); T4dTm3 = T4dTm3[..., np.newaxis, np.newaxis]
    T4dTm3 = tf.convert_to_tensor(T4dTm3, dtype='float32')
    return T4dTm3
def init_weights_T4dMi9(shape, dtype=None):
    scale = 1/75
    T4dMi9 = scale*np.array([[-8,-16, -8],[0, -6, 0],[0, 0, 0]])
    T4dMi9 = T4dMi9[..., np.newaxis, np.newaxis]
    T4dMi9 = tf.convert_to_tensor(T4dMi9, dtype='float32')
    return T4dMi9
def init_weights_T4dMi4(shape, dtype=None):
    scale = 1/75
    T4dMi4 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -8, 0]]); T4dMi4 = T4dMi4[..., np.newaxis, np.newaxis]
    T4dMi4 = tf.convert_to_tensor(T4dMi4, dtype='float32')
    return T4dMi4
def init_weights_T4dC3(shape, dtype=None):
    scale = 1/75
    T4dC3 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -6, 0]]); T4dC3 = T4dC3[..., np.newaxis, np.newaxis]
    T4dC3 = tf.convert_to_tensor(T4dC3, dtype='float32')
    return T4dC3
# LOBULA INITIALIZATION FILTERS
# T5a
def init_weights_T5aTm1(shape, dtype=None):
    scale = 1/75
    T5aTm1 = scale*np.array([[8,8, 0],[8, 32, 0],[8, 24, 0]]); T5aTm1 = T5aTm1[..., np.newaxis, np.newaxis]
    T5aTm1 = tf.convert_to_tensor(T5aTm1, dtype='float32')
    return T5aTm1
def init_weights_T5aTm2(shape, dtype=None):
    scale = 1/75
    T5aTm2 = scale*np.array([[-4,0, 0],[-6, 0, 0],[-8, 0, 0]]); T5aTm2 = T5aTm2[..., np.newaxis, np.newaxis]
    T5aTm2 = tf.convert_to_tensor(T5aTm2, dtype='float32')
    return T5aTm2
def init_weights_T5aTm4(shape, dtype=None):
    scale = 1/75
    T5aTm4 = scale*np.array([[0,0,0,0,0],[0,0,0,-8,0],[0,0,0,-8,-4], [0,0,0,-6,0], [0,0,0,0,0]])
    T5aTm4 = T5aTm4[..., np.newaxis, np.newaxis]
    T5aTm4 = tf.convert_to_tensor(T5aTm4, dtype='float32')
    return T5aTm4
def init_weights_T5aTm9(shape, dtype=None):
    scale = 1/75
    T5aTm9 = scale*np.array([[8,0, 8],[0, 10, 0],[0, 0, 0]]); T5aTm9 = T5aTm9[..., np.newaxis, np.newaxis]
    T5aTm9 = tf.convert_to_tensor(T5aTm9, dtype='float32')
    return T5aTm9
# T5b
def init_weights_T5bTm1(shape, dtype=None):
    scale = 1/75
    T5bTm1 = scale*np.array([[0,8, 8],[0, 32, 8],[0, 8, 8]]); T5bTm1 = T5bTm1[..., np.newaxis, np.newaxis]
    T5bTm1 = tf.convert_to_tensor(T5bTm1, dtype='float32')
    return T5bTm1
def init_weights_T5bTm2(shape, dtype=None):
    scale = 1/75
    T5bTm2 = scale*np.array([[0,0, -8],[0, 0, -8],[0, 0, -8]]); T5bTm2 = T5bTm2[..., np.newaxis, np.newaxis]
    T5bTm2 = tf.convert_to_tensor(T5bTm2, dtype='float32')
    return T5bTm2
def init_weights_T5bTm4(shape, dtype=None):
    scale = 1/75
    T5bTm4 = scale*np.array([[0,0,0,0,0],[0,-16,0,0,0],[-8,-16,0,0,0], [0,-16,0,0,0], [0,0,0,0,0]])
    T5bTm4 = T5bTm4[..., np.newaxis, np.newaxis]
    T5bTm4 = tf.convert_to_tensor(T5bTm4, dtype='float32')
    return T5bTm4
def init_weights_T5bTm9(shape, dtype=None):
    scale = 1/75
    T5bTm9 = scale*np.array([[0,0, 0],[0, 0, 0],[0, 0, 8]]); T5bTm9 = T5bTm9[..., np.newaxis, np.newaxis]
    T5bTm9 = tf.convert_to_tensor(T5bTm9, dtype='float32')
    return T5bTm9
# T5c
def init_weights_T5cTm1(shape, dtype=None):
    scale = 1/75
    T5cTm1 = scale*np.array([[10,8, 16],[8, 32, 0],[6, 0, 0]]); T5cTm1 = T5cTm1[..., np.newaxis, np.newaxis]
    T5cTm1 = tf.convert_to_tensor(T5cTm1, dtype='float32')
    return T5cTm1
def init_weights_T5cTm2(shape, dtype=None):
    scale = 1/75
    T5cTm2 = scale*np.array([[0,-6, 0],[0, 0, 0],[0, 0, 0]]); T5cTm2 = T5cTm2[..., np.newaxis, np.newaxis]
    T5cTm2 = tf.convert_to_tensor(T5cTm2, dtype='float32')
    return T5cTm2
def init_weights_T5cTm4(shape, dtype=None):
    scale = 1/75
    T5cTm4 = scale*np.array([[0,0, 0],[0, -6, 0],[-8, -6, 0]])
    T5cTm4 = T5cTm4[..., np.newaxis, np.newaxis]
    T5cTm4 = tf.convert_to_tensor(T5cTm4, dtype='float32')
    return T5cTm4
def init_weights_T5cTm9(shape, dtype=None):
    scale = 1/75
    T5cTm9 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]]); T5cTm9 = T5cTm9[..., np.newaxis, np.newaxis]
    T5cTm9 = tf.convert_to_tensor(T5cTm9, dtype='float32')
    return T5cTm9
# T5d
def init_weights_T5dTm1(shape, dtype=None):
    scale = 1/75
    T5dTm1 = scale*np.array([[8,0, 0],[8, 32, 0],[8, 8, 10]]); T5dTm1 = T5dTm1[..., np.newaxis, np.newaxis]
    T5dTm1 = tf.convert_to_tensor(T5dTm1, dtype='float32')
    return T5dTm1
def init_weights_T5dTm2(shape, dtype=None):
    scale = 1/75
    T5dTm2 = scale*np.array([[0,0, 0],[0, 0, 0],[0, -8, 0]]); T5dTm2 = T5dTm2[..., np.newaxis, np.newaxis]
    T5dTm2 = tf.convert_to_tensor(T5dTm2, dtype='float32')
    return T5dTm2
def init_weights_T5dTm4(shape, dtype=None):
    scale = 1/75
    T5dTm4 = scale*np.array([[-8,-16, -8],[0, -6, 0],[0, 0, 0]])
    T5dTm4 = T5dTm4[..., np.newaxis, np.newaxis]
    T5dTm4 = tf.convert_to_tensor(T5dTm4, dtype='float32')
    return T5dTm4
def init_weights_T5dTm9(shape, dtype=None):
    scale = 1/75
    T5dTm9 = scale*np.array([[0,8, 0],[0, 10, 0],[0, 8, 0]]); T5dTm9 = T5dTm9[..., np.newaxis, np.newaxis]
    T5dTm9 = tf.convert_to_tensor(T5dTm9, dtype='float32')
    return T5dTm9
# OPTIC GLOMERULI INITIALIZATION FILTERS
# LPLC2T4     Weights from Neuprint
def init_weights_LPLC2T4a(shape, dtype=None):
    scale = 1/75
    LPLC2T4a = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T4a = LPLC2T4a[..., np.newaxis, np.newaxis]
    LPLC2T4a = tf.convert_to_tensor(LPLC2T4a, dtype='float32')
    return LPLC2T4a
def init_weights_LPLC2T4b(shape, dtype=None):
    scale = 1/75
    LPLC2T4b = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T4b = LPLC2T4b[..., np.newaxis, np.newaxis]
    LPLC2T4b = tf.convert_to_tensor(LPLC2T4b, dtype='float32')
    return LPLC2T4b
def init_weights_LPLC2T4c(shape, dtype=None):
    scale = 1/75
    LPLC2T4c = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T4c = LPLC2T4c[..., np.newaxis, np.newaxis]
    LPLC2T4c = tf.convert_to_tensor(LPLC2T4c, dtype='float32')
    return LPLC2T4c
def init_weights_LPLC2T4d(shape, dtype=None):
    scale = 1/75
    LPLC2T4d = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T4d = LPLC2T4d[..., np.newaxis, np.newaxis]
    LPLC2T4d = tf.convert_to_tensor(LPLC2T4d, dtype='float32')
    return LPLC2T4d
# LPLC2T5     Weights from Neuprint
def init_weights_LPLC2T5a(shape, dtype=None):
    scale = 1/75
    LPLC2T5a = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T5a = LPLC2T5a[..., np.newaxis, np.newaxis]
    LPLC2T5a = tf.convert_to_tensor(LPLC2T5a, dtype='float32')
    return LPLC2T5a
def init_weights_LPLC2T5b(shape, dtype=None):
    scale = 1/75
    LPLC2T5b = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T5b = LPLC2T5b[..., np.newaxis, np.newaxis]
    LPLC2T5b = tf.convert_to_tensor(LPLC2T5b, dtype='float32')
    return LPLC2T5b
def init_weights_LPLC2T5c(shape, dtype=None):
    scale = 1/75
    LPLC2T5c = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T5c = LPLC2T5c[..., np.newaxis, np.newaxis]
    LPLC2T5c = tf.convert_to_tensor(LPLC2T5c, dtype='float32')
    return LPLC2T5c
def init_weights_LPLC2T5d(shape, dtype=None):
    scale = 1/75
    LPLC2T5d = scale*np.array([[0,0, 0],[0, 27, 0],[0, 0, 0]]); LPLC2T5d = LPLC2T5d[..., np.newaxis, np.newaxis]
    LPLC2T5d = tf.convert_to_tensor(LPLC2T5d, dtype='float32')
    return LPLC2T5d
####################################
###### Dataset preparation #########
test_set = ['rollerblade', 'scooter-black','scooter-gray', 'soapbox', 'soccerball',
            'stroller', 'surf', 'swing', 'tennis', 'train']
data = read_mat('.\\data\\DAVIS_CNNRNN_data.mat')
print(data.keys())
# Training data
pos_x = np.array([]); pos_y = np.array([]); pos_z = np.array([])
delta_x = np.array([]); delta_y = np.array([]); delta_z = np.array([]); fr_timed = []
for i in range(len(data['training_data'])):
    if any(ele in data['training_data'][i]['label'] for ele in test_set)==False:  # Excluding test data from training data
        if i==0:
            input_frames = data['training_data'][i]['images']
        else:
            input_frames = np.concatenate((input_frames,data['training_data'][i]['images']), axis=0)
        for j in range(data['training_data'][i]['images'].shape[0]-10):
            fr_timed.append(data['training_data'][i]['images'][j:j+10,:,:])
        pos_x = np.append(pos_x,[data['training_data'][i]['x'][0:-1-9]])
        pos_y = np.append(pos_y,[data['training_data'][i]['y'][0:-1-9]])
        pos_z = np.append(pos_z,[data['training_data'][i]['z'][0:-1-9]])
        delta_x = np.append(delta_x, [data['training_data'][i]['delta_x'][0:-1-9]])
        delta_y = np.append(delta_y, [data['training_data'][i]['delta_y'][0:-1-9]])
        delta_z = np.append(delta_z, [data['training_data'][i]['delta_z'][0:-1-9]])
timed_fr = np.array(fr_timed)
print('Frames with time dimension', timed_fr.shape)
print('size of frames', input_frames.shape, 'size of x', pos_x.shape, 'size of y', pos_y.shape, 'size of delta_x',
      delta_x.shape, 'size of delta_y', delta_y.shape)
y_true = np.stack((pos_x, pos_y, pos_z), axis=1); print('Array of true outputs', y_true.shape)
# Validation data
val_pos_x = np.array([]); val_pos_y = np.array([]); val_pos_z = np.array([]); check = 0
val_delta_x = np.array([]); val_delta_y = np.array([]); val_delta_z = np.array([]); val_fr_timed = []
for i in range(len(data['training_data'])):
    if any(ele in data['training_data'][i]['label'] for ele in test_set)==True:
        if check ==0:
            val_input_frames = data['training_data'][i]['images']
            check += 1
        else:
            val_input_frames = np.concatenate((val_input_frames,data['training_data'][i]['images'][0:40,:,:]), axis=0)
        for j in range(data['training_data'][i]['images'].shape[0]-10):
            val_fr_timed.append(data['training_data'][i]['images'][j:j+10,:,:])
        val_pos_x = np.append(val_pos_x,[data['training_data'][i]['x'][0:-1-9]])
        val_pos_y = np.append(val_pos_y,[data['training_data'][i]['y'][0:-1-9]])
        val_pos_z = np.append(val_pos_z,[data['training_data'][i]['z'][0:-1-9]])
        val_delta_x = np.append(val_delta_x, [data['training_data'][i]['delta_x'][0:-1-9]])
        val_delta_y = np.append(val_delta_y, [data['training_data'][i]['delta_y'][0:-1-9]])
        val_delta_z = np.append(val_delta_z, [data['training_data'][i]['delta_z'][0:-1-9]])
val_timed_fr = np.array(val_fr_timed)
print('Frames with time dimension', val_timed_fr.shape)
print('size of frames', val_input_frames.shape, 'size of x', val_pos_x.shape, 'size of y', val_pos_y.shape, 'size of delta_x',
      val_delta_x.shape, 'size of delta_y', val_delta_y.shape)
val_y_true = np.stack((val_pos_x, val_pos_y, val_pos_z), axis=1)
print('Array of true outputs', val_y_true.shape)
####################################
###### CNN model ###################
bias_init = 3.5/75
cnn_model = K.models.Sequential()
inputs = K.Input(shape=[timed_fr.shape[1], timed_fr.shape[2], timed_fr.shape[3], 1])     # (elevation, azimuth, channels)
# LAMINA
L1R = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_L1R,
                         bias_initializer=K.initializers.Constant(bias_init),
                         padding='same', activation='relu'), name='L1R')(inputs)
L2R = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_L2R,
                         bias_initializer=K.initializers.Constant(bias_init),
                         padding='same', activation='relu'), name='L2R')(inputs)
L3R = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_L3R,
                         bias_initializer=K.initializers.Constant(bias_init),
                         padding='same', activation='relu'),name='L3R')(inputs)
L5L1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_L5L1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'),name='L5L1')(L1R)
L5L2 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_L5L2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'),name='L5L2')(L2R)
L5 = K.layers.Add(name='L5')([L5L1, L5L2]); L5 = K.layers.Activation('relu')(L5)
# Outer MEDULLA
Mi1L1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Mi1L1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Mi1L1')(L1R)
Mi1L5 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Mi1L5,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Mi1L5')(L5)
Mi1 = K.layers.Add(name='Mi1')([Mi1L1, Mi1L5]); Mi1 = K.layers.Activation('relu')(Mi1)
Tm1L2 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm1L2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='Tm1L2')(L2R)
Tm2L2 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm2L2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='Tm2L2')(L2R)
Tm3L1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm3L1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Tm3L1')(L1R)
Tm3L5 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm3L5,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Tm3L5')(L5)
Tm3 = K.layers.Add(name='Tm3')([Tm3L1, Tm3L5]); Tm3 = K.layers.Activation('relu')(Tm3)
Tm4L2 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm4L2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='Tm4L2')(L2R)
Mi9L3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Mi9L3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='Mi9L3')(L3R)
Mi4L5 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Mi4L5,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='Mi4L5')(L5)
C3L1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_C3L1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', activation='relu'), name='C3L1')(L1R)
Tm9L3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm9L3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Tm9L3')(L3R)
Tm9Mi4 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_Tm9Mi4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='Tm9Mi4')(Mi4L5)
Tm9 = K.layers.Add(name='Tm9')([Tm9L3, Tm9Mi4]); Tm9 = K.layers.Activation('relu')(Tm9)
# Inner MEDULLA
# T4a
T4aMi1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4aMi1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4aMi1')(Mi1)
T4aMi1_out = T(K.layers.Flatten())(T4aMi1)
T4aMi1_out, T4aMi1_st, T4aMi1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4aMi1_out)
T4aTm3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4aTm3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4aTm3')(Tm3)
T4aTm3_out = T(K.layers.Flatten())(T4aTm3)
T4aTm3_out, T4aTm3_st, T4aTm3_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4aTm3_out)
T4aMi9 = T(K.layers.Conv2D(1,5, use_bias=True, kernel_initializer=init_weights_T4aMi9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4aMi9')(Mi9L3)
T4aMi9_out = T(K.layers.Flatten())(T4aMi9)
T4aMi9_out, T4aMi9_st, T4aMi9_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4aMi9_out)
T4aMi9_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4aMi9_out)
T4aMi4 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4aMi4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4aMi4')(Mi4L5)
T4aMi4_out = T(K.layers.Flatten())(T4aMi4)
T4aMi4_out, T4aMi4_st, T4aMi4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4aMi4_out)
T4aMi4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4aMi4_out)
T4aC3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4aC3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4aC3')(C3L1)
T4aC3_out = T(K.layers.Flatten())(T4aC3)
T4aC3_out, T4aC3_st, T4aC3_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4aC3_out)
T4aC3_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4aC3_out)
T4a = K.layers.Add(name='T4a')([T4aMi1_out, T4aTm3_out, T4aMi9_last, T4aMi4_last, T4aC3_last])
T4a = K.layers.Activation('relu')(T4a)
T4a = K.layers.Reshape((20,35,1))(T4a)
# T4b
T4bMi1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4bMi1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4bMi1')(Mi1)
T4bMi1_out = T(K.layers.Flatten())(T4bMi1)
T4bMi1_out, T4bMi1_st, T4bMi1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4bMi1_out)
T4bTm3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4bTm3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4bTm3')(Tm3)
T4bTm3_out = T(K.layers.Flatten())(T4bTm3)
T4bTm3_out, T4bTm3_st, T4bTm3_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4bTm3_out)
T4bMi9 = T(K.layers.Conv2D(1,5, use_bias=True, kernel_initializer=init_weights_T4bMi9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4bMi9')(Mi9L3)
T4bMi9_out = T(K.layers.Flatten())(T4bMi9)
T4bMi9_out, T4bMi9_st, T4bMi9_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4bMi9_out)
T4bMi9_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4bMi9_out)
T4bMi4 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4bMi4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4bMi4')(Mi4L5)
T4bMi4_out = T(K.layers.Flatten())(T4bMi4)
T4bMi4_out, T4bMi4_st, T4bMi4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4bMi4_out)
T4bMi4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4bMi4_out)
T4bC3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4bC3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4bC3')(C3L1)
T4bC3_out = T(K.layers.Flatten())(T4bC3)
T4bC3_out, T4bC3_st, T4bC3_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4bC3_out)
T4bC3_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4bC3_out)
T4b = K.layers.Add(name='T4b')([T4bMi1_out, T4bTm3_out, T4bMi9_last, T4bMi4_last, T4bC3_last])
T4b = K.layers.Activation('relu')(T4b)
T4b = K.layers.Reshape((20,35,1))(T4b)
# # T4c
T4cMi1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4cMi1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4cMi1')(Mi1)
T4cMi1_out = T(K.layers.Flatten())(T4cMi1)
T4cMi1_out, T4cMi1_st, T4cMi1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4cMi1_out)
T4cTm3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4cTm3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4cTm3')(Tm3)
T4cTm3_out = T(K.layers.Flatten())(T4cTm3)
T4cTm3_out, T4cTm3_st, T4cTm3_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4cTm3_out)
T4cMi9 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4cMi9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4cMi9')(Mi9L3)
T4cMi9_out = T(K.layers.Flatten())(T4cMi9)
T4cMi9_out, T4cMi9_st, T4cMi9_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4cMi9_out)
T4cMi9_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4cMi9_out)
T4cMi4 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4cMi4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4cMi4')(Mi4L5)
T4cMi4_out = T(K.layers.Flatten())(T4cMi4)
T4cMi4_out, T4cMi4_st, T4cMi4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4cMi4_out)
T4cMi4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4cMi4_out)
T4cC3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4cC3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4cC3')(C3L1)
T4cC3_out = T(K.layers.Flatten())(T4cC3)
T4cC3_out, T4cC3_st, T4cC3_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4cC3_out)
T4cC3_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4cC3_out)
T4c = K.layers.Add(name='T4c')([T4cMi1_out, T4cTm3_out, T4cMi9_last, T4cMi4_last, T4cC3_last])
T4c = K.layers.Activation('relu')(T4c)
T4c = K.layers.Reshape((20,35,1))(T4c)
# # T4d
T4dMi1 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4dMi1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4dMi1')(Mi1)
T4dMi1_out = T(K.layers.Flatten())(T4dMi1)
T4dMi1_out, T4dMi1_st, T4dMi1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4dMi1_out)
T4dTm3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4dTm3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4dTm3')(Tm3)
T4dTm3_out = T(K.layers.Flatten())(T4dTm3)
T4dTm3_out, T4dTm3_st, T4dTm3_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T4dTm3_out)
T4dMi9 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4dMi9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4dMi9')(Mi9L3)
T4dMi9_out = T(K.layers.Flatten())(T4dMi9)
T4dMi9_out, T4dMi9_st, T4dMi9_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4dMi9_out)
T4dMi9_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4dMi9_out)
T4dMi4 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4dMi4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4dMi4')(Mi4L5)
T4dMi4_out = T(K.layers.Flatten())(T4dMi4)
T4dMi4_out, T4dMi4_st, T4dMi4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4dMi4_out)
T4dMi4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4dMi4_out)
T4dC3 = T(K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T4dC3,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same'), name='T4dC3')(C3L1)
T4dC3_out = T(K.layers.Flatten())(T4dC3)
T4dC3_out, T4dC3_st, T4dC3_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T4dC3_out)
T4dC3_last = K.layers.Lambda(lambda x: x[:, 8, :])(T4dC3_out)
T4d = K.layers.Add(name='T4d')([T4dMi1_out, T4dTm3_out, T4dMi9_last, T4dMi4_last, T4dC3_last])
T4d = K.layers.Activation('relu')(T4d)
T4d = K.layers.Reshape((20,35,1))(T4d)
# # LOBULA
# # T5a
T5aTm1 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5aTm1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5aTm1')(Tm1L2)
T5aTm1_out = T(K.layers.Flatten())(T5aTm1)
T5aTm1_out, T5aTm1_st, T5aTm1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5aTm1_out)
T5aTm2 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5aTm2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5aTm2')(Tm2L2)
T5aTm2_out = T(K.layers.Flatten())(T5aTm2)
T5aTm2_out, T5aTm2_st, T5aTm2_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5aTm2_out)
T5aTm2_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5aTm2_out)
T5aTm4 = K.layers.Conv2D(1,5, use_bias=True, kernel_initializer=init_weights_T5aTm4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5aTm4')(Tm4L2)
T5aTm4_out = T(K.layers.Flatten())(T5aTm4)
T5aTm4_out, T5aTm4_st, T5aTm4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5aTm4_out)
T5aTm4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5aTm4_out)
T5aTm9 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5aTm9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5aTm9')(Tm9)
T5aTm9_out = T(K.layers.Flatten())(T5aTm9)
T5aTm9_out, T5aTm9_st, T5aTm9_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5aTm9_out)
T5a = K.layers.Add(name='T5a')([T5aTm1_out, T5aTm2_last, T5aTm4_last, T5aTm9_out])
T5a = K.layers.Activation('relu')(T5a)
T5a = K.layers.Reshape((20,35,1))(T5a)
# # T5b
T5bTm1 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5bTm1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5bTm1')(Tm1L2)
T5bTm1_out = T(K.layers.Flatten())(T5bTm1)
T5bTm1_out, T5bTm1_st, T5bTm1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5bTm1_out)
T5bTm2 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5bTm2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5bTm2')(Tm2L2)
T5bTm2_out = T(K.layers.Flatten())(T5bTm2)
T5bTm2_out, T5bTm2_st, T5bTm2_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5bTm2_out)
T5bTm2_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5bTm2_out)
T5bTm4 = K.layers.Conv2D(1,5, use_bias=True, kernel_initializer=init_weights_T5bTm4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5bTm4')(Tm4L2)
T5bTm4_out = T(K.layers.Flatten())(T5bTm4)
T5bTm4_out, T5bTm4_st, T5bTm4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5bTm4_out)
T5bTm4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5bTm4_out)
T5bTm9 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5bTm9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5bTm9')(Tm9)
T5bTm9_out = T(K.layers.Flatten())(T5bTm9)
T5bTm9_out, T5bTm9_st, T5bTm9_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5bTm9_out)
T5b = K.layers.Add(name='T5b')([T5bTm1_out, T5bTm2_last, T5bTm4_last, T5bTm9_out])
T5b = K.layers.Activation('relu')(T5b)
T5b = K.layers.Reshape((20,35,1))(T5b)
# # T5c
T5cTm1 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5cTm1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5cTm1')(Tm1L2)
T5cTm1_out = T(K.layers.Flatten())(T5cTm1)
T5cTm1_out, T5cTm1_st, T5cTm1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5cTm1_out)
T5cTm2 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5cTm2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5cTm2')(Tm2L2)
T5cTm2_out = T(K.layers.Flatten())(T5cTm2)
T5cTm2_out, T5cTm2_st, T5cTm2_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5cTm2_out)
T5cTm2_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5cTm2_out)
T5cTm4 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5cTm4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5cTm4')(Tm4L2)
T5cTm4_out = T(K.layers.Flatten())(T5cTm4)
T5cTm4_out, T5cTm4_st, T5cTm4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5cTm4_out)
T5cTm4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5cTm4_out)
T5cTm9 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5cTm9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5cTm9')(Tm9)
T5cTm9_out = T(K.layers.Flatten())(T5cTm9)
T5cTm9_out, T5cTm9_st, T5cTm9_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5cTm9_out)
T5c = K.layers.Add(name='T5c')([T5cTm1_out, T5cTm2_last, T5cTm4_last, T5cTm9_out])
T5c = K.layers.Activation('relu')(T5c)
T5c = K.layers.Reshape((20,35,1))(T5c)
# # T5d
T5dTm1 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5dTm1,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5dTm1')(Tm1L2)
T5dTm1_out = T(K.layers.Flatten())(T5dTm1)
T5dTm1_out, T5dTm1_st, T5dTm1_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5dTm1_out)
T5dTm2 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5dTm2,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5dTm2')(Tm2L2)
T5dTm2_out = T(K.layers.Flatten())(T5dTm2)
T5dTm2_out, T5dTm2_st, T5dTm2_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5dTm2_out)
T5dTm2_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5dTm2_out)
T5dTm4 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5dTm4,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5dTm4')(Tm4L2)
T5dTm4_out = T(K.layers.Flatten())(T5dTm4)
T5dTm4_out, T5dTm4_st, T5dTm4_del = K.layers.LSTM(units=700, return_sequences=True, return_state=True, trainable=False)(T5dTm4_out)
T5dTm4_last = K.layers.Lambda(lambda x: x[:, 8, :])(T5dTm4_out)
T5dTm9 = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_T5dTm9,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='T5dTm9')(Tm9)
T5dTm9_out = T(K.layers.Flatten())(T5dTm9)
T5dTm9_out, T5dTm9_st, T5dTm9_del = K.layers.LSTM(units=700, return_state=True, trainable=False)(T5dTm9_out)
T5d = K.layers.Add(name='T5d')([T5dTm1_out, T5dTm2_last, T5dTm4_last, T5dTm9_out])
T5d = K.layers.Activation('relu')(T5d)
T5d = K.layers.Reshape((20,35,1))(T5d)
# # OPTIC GLOMERULI
# # LPLC2T4
LPLC2T4a = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T4a,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T4a')(T4a)
LPLC2T4b = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T4b,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T4b')(T4b)
LPLC2T4c = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T4c,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T4c')(T4c)
LPLC2T4d = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T4d,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T4d')(T4d)
# # LPLC2T5
LPLC2T5a = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T5a,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T5a')(T5a)
LPLC2T5b = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T5b,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T5b')(T5b)
LPLC2T5c = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T5c,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T5c')(T5c)
LPLC2T5d = K.layers.Conv2D(1,3, use_bias=True, kernel_initializer=init_weights_LPLC2T5d,
                        bias_initializer=K.initializers.Constant(bias_init),
                       padding='same', name='LPLC2T5d')(T5d)
LPLC2 = K.layers.Add(name='LPLC2')([LPLC2T4a, LPLC2T4b, LPLC2T4c, LPLC2T4d, LPLC2T5a, LPLC2T5b, LPLC2T5c, LPLC2T5d])
LPLC2 = K.layers.Activation('relu')(LPLC2)
# FULLY CONNECTED LAYERS
FC = K.layers.Concatenate(axis=-1)([T4a, T4b, T4c, T4d, T5a, T5b, T5c, T5d, LPLC2])
FC = K.layers.Flatten()(FC)
FC = K.layers.Dense(128, kernel_initializer='normal', kernel_regularizer=l2(1e-3),
                    activity_regularizer=l1(1e-3), activation='relu')(FC)
FC = K.layers.Dense(32, kernel_initializer='normal', kernel_regularizer=l2(1e-3),
                    activity_regularizer=l1(1e-3), activation='relu')(FC)
outputs = K.layers.Dense(3, kernel_initializer='normal', kernel_regularizer=l2(1e-3),
                         activity_regularizer=l1(1e-3), activation='linear')(FC)
cnn_model = K.Model(inputs=inputs, outputs=outputs, name='cnn_model')
print(cnn_model.summary())
K.utils.plot_model(cnn_model, '.\\images\\CONNECTOME_CNN_RNN_DIAGRAM.png')
####################################
###### Training ####################
# Loss function and optimizer algorithm
lr = 1e-3; bz = 40; nb_epochs = 130; val_split = 0.10    # percentage of training data as validation data
cnn_model.compile(loss='MSE', optimizer=Adam(lr), metrics=['accuracy'])
# define model callbacks
cbs = [cb.EarlyStopping(monitor='val_loss', min_delta=0.2, patience=15),
       TensorBoard(log_dir="logs/{}".format(time()))]
# train
history = cnn_model.fit(timed_fr, y_true, batch_size=bz, epochs=nb_epochs,
                        validation_data=(val_timed_fr, val_y_true), callbacks=cbs)
cnn_model.save('connectome_model_CNNRNN_v3')
####################################
###### Plotting ####################
fig1, ax1 = plt.subplots(2,1)
ax1[0].plot(history.history['accuracy'])
ax1[0].set_title('model accuracy'); ax1[0].set_ylabel('accuracy'); ax1[0].set_xlabel('epoch')
ax1[0].legend(['train_accuracy', 'val_accuracy'], loc='upper right', frameon=False)
ax1[1].plot(history.history['loss']); ax1[1].plot(history.history['val_loss'])
ax1[1].set_title('model loss'); ax1[1].set_ylabel('loss'); ax1[1].set_xlabel('epoch')
ax1[1].legend(['train_loss', 'val_loss'], loc='upper right', frameon=False)

predictions = cnn_model.predict(timed_fr)
print(predictions.shape)

fig2, ax2 = plt.subplots(3); idd_l = np.array([0,0,0,1,1,1]); idd_r = np.array([0,1,2,0,1,2])
bar_dir = ['x','y','z','delta_x','delta_y','delta_z']
fig2.suptitle('Training data prediction')
for i in range(predictions.shape[1]):
    ax2[i].plot(np.arange(0,1000), y_true[:1000,i], linewidth=1, color='black', alpha=0.7)
    ax2[i].plot(np.arange(0,1000), predictions[:1000,i], linewidth=1, color='blue')
    ax2[i].legend(['ground truth', 'prediction'], loc='upper right', frameon=False)
    ax2[i].set_title('{bar_dir}'.format(bar_dir=bar_dir[i]))
plt.show()
####################################
