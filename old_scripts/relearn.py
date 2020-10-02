#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import load_model
import datetime


epsilon = K.epsilon()
from io import BytesIO, StringIO
from tensorflow.python.lib.io import file_io
import argparse

import keras
global max_epochs


flag_show_plots = True # True for Notebooks, False otherwise
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

from keras.models import Model
from keras.layers import Flatten, Input, Dense , Bidirectional , ConvLSTM2D , concatenate , Reshape , CuDNNGRU , subtract , SpatialDropout2D

from keras import backend
from keras.callbacks import Callback as Callback
from keras.backend import tensorflow_backend as K

import tensorflow as tf


################################################################################
dirlocal = './'
dataset = 'full' # 'sample' or 'full'
stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')

modelfile = 'model-' + str(stamp) + '.h5'

max_epochs = 64
es_patience = 100

if dataset == 'sample':
    max_epochs = 8
    es_patience = 1


# In[2]:



################################################################################
def determine_number_of_channels(input_features, pdb_list, length_dict):
    F = 0
    rows = 0
    channels = 0
    x = input_features[pdb_list[0]]
    l = length_dict[pdb_list[0]]
    for feature in x:
        if len(feature) == l:
            F += 2
            rows +=1
        elif len(feature) == l * l:
            F += 1
            channels +=1
        else:
            print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
            sys.exit(1)
    return F , rows, channels


################################################################################
def print_max_avg_sum_of_each_channel(x):
    print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i, a, m, s))

################################################################################
# Roll out 1D features to two 2D features, all to 256 x 256 (because many are smaller)
def prepare_input_features_2D(pdbs, input_features, distance_maps_cb, length_dict, F, fft = False):


    Y = np.full((len(pdbs), 256, 256, 1), 100.0)




    X = np.full((len(pdbs), 256, 256, F), 0.0)

    for i, pdb in enumerate(pdbs):
        x = input_features[pdb]
        y = distance_maps_cb[pdb]
        l = length_dict[pdb]
        newi = 0
        xmini = np.zeros((l, l, F))
        for feature in x:
            feature = np.array(feature)
            feature = feature.astype(np.float)
            if len(feature) == l:
                for k in range(0, l):
                    xmini[k, :, newi] = feature
                    xmini[:, k, newi + 1] = feature
                newi += 2
            elif len(feature) == l * l:
                xmini[:, :, newi] = feature.reshape(l, l)
                newi += 1
            else:
                print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
                sys.exit(1)
        if l > 256:
            l = 256
        X[i, 0:l, 0:l, :] = xmini[:l, :l, :]
        Y[i, 0:l, 0:l, 0] = y[:l, :l]
    return X, Y

def prepare_input_features_2D_w_1D(pdbs, input_features, distance_maps_cb, length_dict, rownum, channelnum , maxseqs = 5000 , fft = False ):
    #keep features in their native shape...
    nsamples = min( len(pdbs) , maxseqs )
    print(nsamples)
    print(rownum)
    Y = np.full( (nsamples , 256,  256) ,  100.0 , dtype= np.float64  )

    Xrows = np.full((nsamples,  256 , rownum), 0.0  )
    Xchannels = np.full((nsamples, 256, 256, channelnum  ), 0.0 )
    for i, pdb in enumerate(pdbs[0:nsamples]) :
        channels = []
        rows = []



        x = input_features[pdb]
        y = distance_maps_cb[pdb]
        l = length_dict[pdb]
        newi = 0
        xmini = np.zeros((l, l, rownum))
        for feature in x:
            feature = np.array(feature)
            feature = feature.astype(np.float)
            if len(feature) == l:
                #add sequence feature vectors to rnn input matrix
                rows.append(feature)
            elif len(feature) == l * l:
                #add to cnn input channels
                channels.append(feature.reshape(l, l))
            else:
                print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
                sys.exit(1)
        if l > 256:
            l = 256

        xrowsmini = np.vstack(rows).T
        xchannelsmini = np.stack(channels , axis = 2 )


        #so we're just going to ignore anything past 256 aa for folding?

        Xrows[i, 0:l , :] = xrowsmini[:l, : ]
        Xchannels[i, 0:l, 0:l, :] = xchannelsmini[:l, :l, :]
        Y[i, 0:l , 0:l] = y

    return np.array(Xrows), np.array(Xchannels) , Y


################################################################################


def plot_input_output_of_this_protein(Xr, Xc, Y):
    figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    for i in range( Xc.shape[2] ):
        plt.subplot(7, 7, i + 1)
        plt.grid(None)
        plt.imshow(Xc[:, :, i], cmap='RdYlBu', interpolation='nearest')
    # Last plot is the true distance map


    plt.subplot(7, 7, 13)
    plt.grid(None)
    plt.imshow(Xr, cmap='Spectral', interpolation='nearest')
    plt.show()


    plt.subplot(7, 7, 14)
    plt.grid(None)
    plt.imshow(Y[:, :], cmap='Spectral', interpolation='nearest')
    plt.show()



################################################################################
def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    plot_count = 0
    if flag_show_plots:
        plot_count = 4
    avg_mae = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_mae = 0.0
        for pair in top_pairs:
            abs_dist = abs(y_dict[pair] - p_dict[pair])
            sum_mae += abs_dist
        sum_mae /= L
        avg_mae += sum_mae
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + ' = %.2f' % sum_mae)
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(0, L):
                    if not (j, k) in top_pairs:
                        P[j, k] = np.inf
                        Y[j, k] = np.inf
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average MAE = %.2f' % (avg_mae / len(PRED[:, 0, 0, 0])))

################################################################################

################################################################################
print('')
print('Load input features..')
x = dirlocal + dataset + '-input-features.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-input-features.npy', binary_mode=True))
(pdb_list, length_dict, input_features) = np.load(x, encoding='latin1' , allow_pickle=True)

################################################################################
print('')
print('Load distance maps..')
x = dirlocal + dataset + '-distance-maps-cb.npy.1'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-distance-maps-cb.npy', binary_mode=True))
(pdb_list_y, distance_maps_cb) = np.load(x, encoding='latin1', allow_pickle=True)

################################################################################
print('')
print ('Some cross checks on data loading..')
for pdb in pdb_list:
    if not pdb in pdb_list_y:
        print ('I/O mismatch ', pdb)
        sys.exit(1)

################################################################################
print('')
print('Find the number of input channels..')
F, rows, channels = determine_number_of_channels(input_features, pdb_list, length_dict)

################################################################################
print('')
print('Split into training and validation set (4%)..')
split = int(0.04 * len(pdb_list))
valid_pdbs = pdb_list[:split]
train_pdbs = pdb_list[split:]

print('Total validation proteins = ', len(valid_pdbs))
print('Total training proteins = ', len(train_pdbs))

################################################################################


################################################################################
print('')
print('Load input features..')
x = dirlocal + dataset + '-input-features.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-input-features.npy', binary_mode=True))
(pdb_list, length_dict, input_features) = np.load(x, encoding='latin1' , allow_pickle=True)

################################################################################
print('')
print('Load distance maps..')
x = dirlocal + dataset + '-distance-maps-cb.npy.1'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-distance-maps-cb.npy', binary_mode=True))
(pdb_list_y, distance_maps_cb) = np.load(x, encoding='latin1', allow_pickle=True)

################################################################################
print('')
print ('Some cross checks on data loading..')
for pdb in pdb_list:
    if not pdb in pdb_list_y:
        print ('I/O mismatch ', pdb)
        sys.exit(1)

################################################################################
print('')
print('Find the number of input channels..')
F, rows, channels = determine_number_of_channels(input_features, pdb_list, length_dict)

################################################################################
print('')
print('Split into training and validation set (4%)..')
split = int(0.04 * len(pdb_list))
valid_pdbs = pdb_list[:split]
train_pdbs = pdb_list[split:]

print('Total validation proteins = ', len(valid_pdbs))
print('Total training proteins = ', len(train_pdbs))

################################################################################


# In[4]:



import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler,  Normalizer , MinMaxScaler , RobustScaler
from sklearn.decomposition import PCA


# In[5]:

fft = True
pca = True

N= 3000

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape

        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X



class NDSPCA(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = PCA(copy = True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape

        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        self.explained_variance_ratio_ = self._scaler.explained_variance_ratio_
        self.components_ =self._scaler.components_

        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)

        return X

    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

#pca then scale

print('')
print ('Prepare the validation input and outputs..')
XVALIDrows , XVALIDchannels ,   YVALID = prepare_input_features_2D_w_1D(valid_pdbs, input_features, distance_maps_cb, length_dict, rows, channels , maxseqs = N )
print(XVALIDrows.shape)
print(XVALIDchannels.shape)
print(YVALID.shape)

#YFFTvalid = np.stack([ np.fft.rfft2(YVALID[i,:,:]) for i in range(YVALID.shape[0])] )
#YFFTvalidsplit = np.hstack(  [ np.real(YFFTvalid) , np.imag(YFFTvalid)]  )
YFFTvalidsplit = YVALID
print('')
print ('Prepare the training input and outputs..')
XTRAINrows, XTRAINchannels , YTRAIN = prepare_input_features_2D_w_1D(train_pdbs, input_features, distance_maps_cb, length_dict, rows, channels , maxseqs = N )
print(XTRAINrows.shape)
print(XTRAINchannels.shape)
print(YTRAIN.shape)

#YFFTtrain = np.stack([ np.fft.rfft2(YTRAIN[i,:,:]) for i in range(YTRAIN.shape[0])] )
#YFFTtrainsplit =  np.hstack( [ np.real(YFFTtrain) , np.imag(YFFTtrain)]  )

YFFTtrainsplit = YTRAIN

print('final shape y')
print(YFFTtrainsplit.shape)
print(YFFTvalidsplit.shape)

if pca == True:
    #pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None )
    #pca.fit(YFFTvalidsplit)
    #print(pca.explained_variance_ratio_)
    ndpca = NDSPCA(n_components=200)
    ndpca.fit(YFFTtrainsplit[:1000])
    print(ndpca.explained_variance_ratio_)
    print(np.sum(ndpca.explained_variance_ratio_))
    YFFTvalidsplit = ndpca.transform(YFFTvalidsplit)
    YFFTtrainsplit = ndpca.transform(YFFTtrainsplit)

    scaler0 = RobustScaler( )
    scaler0.fit(YFFTtrainsplit)

    YFFTtrainsplit = scaler0.transform( YFFTtrainsplit)
    YFFTvalidsplit = scaler0.transform( YFFTvalidsplit)

'''
    scaler1 = MinMaxScaler( )
    scaler1.fit(YFFTtrainsplit)

    YFFTtrainsplit = scaler1.transform( YFFTtrainsplit)
    YFFTvalidsplit = scaler1.transform( YFFTvalidsplit)'''
#print(YFFTtrainsplit)


if fft == True:
    YTRAIN = YFFTtrainsplit
    YVALID = YFFTvalidsplit



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
K.set_session(tf.Session(config=config))

#first part of the network, stacked GRU
LSTMoutdim = 5
LSTMlayers = 2

#spectral output
Densegrulayers = 2
Densegruoutdim = 25

#second part of the network, CNN stack
CNNlayers = 2

#third part of the network, dense
Denselayers = 3
Denseoutdim = 50
layeroutputs = {}
retrain = False

LSTMoutdimfinal=10




retrain = False

if retrain == False:
    inputrnn = Input(name='Seqin', shape=( XTRAINrows.shape[1] ,XTRAINrows.shape[2] ) )
    for n in range(LSTMlayers):
        if n == 0:
            layer = CuDNNGRU(LSTMoutdim  ,name='gru_'+str(n) ,  return_sequences=True,
            return_state=False, go_backwards=False, stateful=False )
            layer = Bidirectional(layer, merge_mode='concat', weights=None)
            x = layer(inputrnn)
        else:
            layer = CuDNNGRU(LSTMoutdim ,  name='gru_'+str(n),
            return_sequences=True, return_state=False, go_backwards=False, stateful=False )
            layer = Bidirectional(layer, merge_mode='concat', weights=None)
            x = layer(x)
    layer = CuDNNGRU(LSTMoutdim ,  name='gru_'+str(n+1),
    return_sequences=True, return_state=False, go_backwards=False, stateful=False )
    x = layer(x)

    xf =  Flatten()(x)
    print(xf)

    #Deep subnet for RNN out
    for n in range(Densegrulayers):
        layer = Dense( int(Densegruoutdim) , name = 'Densegru_r_'+str(n) , activation='sigmoid')
        x = layer(x)

    layer = Dense( YTRAIN.shape[1] , name = 'gru2Dense'+str(n) , activation='linear'  )
    ximg = layer(xf)

    #CNN
    CNNin = Input( name='ChannelsIn', shape= (XTRAINchannels.shape[1],XTRAINchannels.shape[2], XTRAINchannels.shape[3] )  )
    for n in range(CNNlayers):
        if n == 0:
            layer = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')
            x = layer(CNNin)
            layer = Convolution2D( 16 , kernel_size= 4 , padding = 'same', activation = 'relu' )
            x = layer(CNNin)
        else:
            layer = Convolution2D( 16 , kernel_size= 3  , padding = 'same', activation = 'relu' )
            x = layer(x)
    layer = Convolution2D(8 , kernel_size= 3, padding = 'same' , activation = 'relu')
    x = layer(x)
    x = Flatten()(x)
    #Final deep net
    x = concatenate([xf , x], axis=-1)

    for n in range(Denselayers):
        layer = Dense( int(Denseoutdim) , name = 'Denser_'+str(n) , activation='sigmoid'  )
        x = layer(x)
    layer = Dense( YTRAIN.shape[1] , activation='linear' ,name = 'outputfinal' )
    output = layer(x)

    with tf.device('/cpu:0'):
        model = Model(inputs = [inputrnn , CNNin ] , outputs = [ximg,output] )

    #o  = keras.optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #o  = keras.optimizers.RMSprop(lr=.1, rho=0.9)
    o = keras.optimizers.Adadelta(lr=1.0, rho=0.95)

    #model.compile( optimizer=o, loss= component_loss , metrics=['mae'])
    model.compile( optimizer=o, loss= 'mae' , metrics=['mae'])

else:
    model = print('Load the model..')
    modelfile = 'model-11_06_2019_13_42_54_831534.h5'
    model = load_model(modelfile, custom_objects= { 'tf' : tf  } )
max_len = 3000
mc = ModelCheckpoint(modelfile, monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 2, patience = es_patience)

history = model.fit( { 'Seqin': XTRAINrows ,'ChannelsIn':XTRAINchannels} , [YTRAIN,YTRAIN] , batch_size = None , verbose = 2 , epochs = 100000 ,  validation_data=([XVALIDrows, XVALIDchannels], [YVALID,YVALID] ), callbacks=[es, mc])




# 3-28-2019
# Badri Adhikari
# https://badriadhikari.github.io/

# This script loads a trained model and predicts (writes) topL long-range distances in the CASP RR format
# See RR format description at http://predictioncenter.org/casp8/index.cgi?page=format#RR

################################################################################
from keras.models import load_model
from keras.layers import *
import numpy as np
import tensorflow as tf
import keras

################################################################################
dirlocal = './'
diroutput = './predictions/'

def prepare_input_features_2D_w_1D(this_input_features, l, F , rownum , channelnum ):
    #keep features in their native shape...
    #nsamples = min( len(pdbs) , maxseqs )
    #Y = np.full((nsamples , 256 * 256 ), 100.0  )
    Xrows = np.full((1,  256 , rownum), 0.0  )
    Xchannels = np.full((1,  256, 256, channelnum  ), 0.0 )

    channels = []
    rows = []

    x = input_features[pdb]
    #y = distance_maps_cb[pdb]
    #l = length_dict[pdb]
    newi = 0
    xmini = np.zeros((l, l, rownum))
    for feature in x:
        feature = np.array(feature)
        feature = feature.astype(np.float)
        if len(feature) == l:
            #add sequence feature vectors to rnn input matrix
            rows.append(feature)
        elif len(feature) == l * l:
            #add to cnn input channels
            channels.append(feature.reshape(l, l))
        else:
            print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
            sys.exit(1)
    if l > 256:
        l = 256

    xrowsmini = np.vstack(rows).T
    xchannelsmini = np.stack(channels , axis = 2 )
    Xrows[0, 0:l , :] = xrowsmini[:l, : ]
    Xchannels[0, 0:l, 0:l, :] = xchannelsmini[:l, :l, :]
    return Xrows, Xchannels


# In[51]:


#predict
################################################################################
model = print('Load the model..')
modelfile = './model-11_04_2019_14_23_59_056489.h5'
model = load_model(modelfile, custom_objects={ 'tf' : tf  })
print('done loading')


# In[52]:



import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

print('load data')
x = dirlocal + 'testset-input-features.npy'
(pdb_list, length_dict, sequences, input_features)  = np.load(x)
x = dirlocal + 'testset-distance-maps-cb.npy'


(pdb_list_y, distance_maps_cb) = np.load(x)
F , rows, channels = determine_number_of_channels(input_features, pdb_list, length_dict)

print('done')
print(rows)
print(channels)


# In[53]:



################################################################################
for pdb in sorted(pdb_list):
    print (pdb)
    xrows,xchannels = prepare_input_features_2D_w_1D(input_features[pdb], length_dict[pdb] , F , rows , channels )
    PREDrnn, PRED = model.predict([xrows,xchannels])
    L = length_dict[pdb]
    if L > 256:
        L = 256

    print(split)
    #inverse transfor from complex
    print(PRED.shape)
    print(PRED)


    #inverse the pca and scaling

    #inv = scaler.inverse_transform(PRED)
    inv = ndpca.inverse_transform(PRED)
    print(inv.shape)
    #split = int( int(inv.shape[1])/2 )


    #img = np.real(np.fft.irfft2( inv[0,0:split,:] + 1j*inv[0,split:,:]  ) )
    img = inv
    #img += img.T
    #img /=2

    print(img.shape)
    #plt.hist(img.ravel)
    #plt.show()
    plt.imshow(img[0,:,:] , cmap='RdYlBu', interpolation='nearest' )
    plt.show()

    P = img
    '''
    with open(diroutput + pdb + '.dmap', 'w') as f:
        for j in range(0, L):
            for k in range(0, L):
                f.write("%.2f " % P[j, k])
            f.write('\n')
        f.close()
        # Write top L predicted long-range distances to a separate file
        SEQSEP = 24
        for j in range(0, L):
            for k in range(0, L):
                if k - j < SEQSEP:
                    P[j, k] = np.inf
        p_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
        x = L
        DEVIATIONPOSNEG = 0.1
        f = open(diroutput + pdb + '.rr', 'w')
        f.write(sequences[pdb] + '\n')
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (i, j) = k
            f.write("%d %d %.2f %.2f %.5f\n" % (i, j, p_dict[k] - DEVIATIONPOSNEG , p_dict[k] + DEVIATIONPOSNEG, 1 / p_dict[k]))
            x -= 1
            if x == 0:
                break
    '''


# In[ ]:




P = model.predict(XVALID)

print('')
print('Compare the predictions with the truths (for some proteins) ..')
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    I = 1
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(4, 4, I)
        I += 1
        plt.grid(None)
        plt.imshow(P[k, 0:L, 0:L, 0], cmap='RdYlBu', interpolation='nearest')
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(4, 4, I)
        I += 1
        plt.grid(None)
        plt.imshow(YVALID[k, 0:L, 0:L, 0], cmap='Spectral', interpolation='nearest')
    plt.show()

################################################################################
print('')
print('MAE of top L long-range distance predictions on the validation set..')
calculate_mae(P, YVALID, valid_pdbs, length_dict)


################################################################################
print('')
print('Evaluate on the test dataset..')
model = load_model(modelfile, compile = False)
x = dirlocal + 'testset-input-features.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-input-features.npy', binary_mode=True))
(pdb_list, length_dict, sequence_dict, input_features)  = np.load(x)
x = dirlocal + 'testset-distance-maps-cb.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-distance-maps-cb.npy', binary_mode=True))
(pdb_list_y, distance_maps_cb) = np.load(x)
F = determine_number_of_channels(input_features, pdb_list, length_dict)
XTEST, YTEST = prepare_input_features_2D(pdb_list, input_features, distance_maps_cb, length_dict, F)
P = model.predict(XTEST)
for pdb in length_dict:
    if length_dict[pdb] > 256:
        length_dict[pdb] = 256

print('')
print('MAE of top L long-range distance predictions on the test set..')
calculate_mae(P, YTEST, pdb_list, length_dict)

################################################################################
main('./')


# In[ ]:





# In[ ]:
