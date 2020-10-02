
# 4-6-2019
# Badri Adhikari
# https://badriadhikari.github.io/
################################################################################

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
from keras.utils import multi_gpu_model

import keras
global max_epochs


flag_show_plots = True # True for Notebooks, False otherwise
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

from keras.models import Model
from keras.layers import Flatten, Input, Dense , Bidirectional , ConvLSTM2D , concatenate , Reshape , CuDNNGRU , subtract

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

def prepare_input_features_2D_w_1D(pdbs, input_features, distance_maps_cb, length_dict, rownum, channelnum , maxseqs = 100 , fft = False ):
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

fft = True

print('')
print ('Prepare the validation input and outputs..')
XVALIDrows , XVALIDchannels ,   YVALID = prepare_input_features_2D_w_1D(valid_pdbs, input_features, distance_maps_cb, length_dict, rows, channels , fft=fft )
print(XVALIDrows.shape)
print(XVALIDchannels.shape)

print(YVALID.shape)


print('')
print ('Prepare the training input and outputs..')
XTRAINrows, XTRAINchannels , YTRAIN = prepare_input_features_2D_w_1D(train_pdbs, input_features, distance_maps_cb, length_dict, rows, channels , fft = fft)
print(XTRAINrows.shape)
print(XTRAINchannels.shape)
print(YTRAIN.shape)




config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
K.set_session(tf.Session(config=config))



#first part of the network, stacked GRU
LSTMoutdim = 5
LSTMlayers = 2

#spectral output
Densegrulayers =2
Densegruoutdim = 100

#second part of the network, CNN stack
CNNlayers = 4

#third part of the network, dense
Denselayers = 4
Denseoutdim = 150
layeroutputs = {}
retrain = False


if retrain == False:
    neg = Lambda( lambda x : subtract(  [tf.zeros_like(x) , x]  ) )
    inputrnn = Input(name='Seqin', shape=( 256,5 ) )
    print(inputrnn)
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
    print(x)
    x = Reshape( (256*LSTMoutdim ) ) 
    print(x)
    #dense in complex
    for n in range(Densegrulayers):
        if n == 0 :
            layer1 = Dense( int(Densegruoutdim) , name = 'Densegru_r_'+str(n) , activation='relu')
            x = layer(x)
            """layer1 = Dense( int(Densegruoutdim/2) , name = 'Densegru_r_'+str(n) , activation='relu')
            layer2 = Dense( int(Densegruoutdim/2) , name = 'Densegru_i_'+str(n) , activation='relu')

            x_r = layer1(x)
            layer = Dropout( .01 )
            x_r = layer(x_r)

            x_i = layer2(x)
            layer = Dropout( .01 )
            x_i = layer(x_i)
            """
        else:
            layer1 = Dense( int(Densegruoutdim) , name = 'Densegru_r_'+str(n) , activation='relu')
            x = layer(x)


            """nx_i = neg(x_i)
            x_1 = concatenate([x_r, nx_i], axis=-1)
            x_2 = concatenate([x_i, x_r], axis=-1)
            print(x_1)
            print(x_2)
            layer = Dense(Densegruoutdim,  name = 'Dense_gru_'+str(n) ,  activation='relu')

            x_r = layer(x_1)
            layer = Dropout( .01 )
            x_r = layer(x_r)

            x_i = layer(x_2)
            layer = Dropout( .01 )
            x_i = layer(x_i)
            """
    """
    nx_i = neg(x_i)
    x_f1 = concatenate([x_r, x_i], axis=-1)
    x_f2 = concatenate([x_i, x_r], axis=-1)

    """
    x_f1 = x
    layer = Dense( YTRAIN.shape[1]*YTRAIN.shape[2] , name = 'gru2Dense'+str(n) , activation='relu')

    """
    x_r = layer(x_f1)
    x_i = layer(x_f2)


    x_r =  Reshape( (256,256) )(x_r)
    x_i =  Reshape( (256,256) )(x_i)

    xcomplex = Lambda(lambda x: tf.complex(x[0], x[1]))([x_r, x_i])
    #layer = Lambda(lambda x :  tf.real( tf.spectral.ifft2d(x)) , output_shape=(256,256) )
    #
    layer =  Lambda(lambda x :  tf.real( x ), output_shape=(256,256) )
    """
    ximg =  Reshape( (256,256) )(x)


    #back to spatial
    CNNin = Input( name='ChannelsIn', shape= ( 256, 256, 3 )  )

    xin = Reshape( (256,256,1))(ximg)

    CNNin = concatenate([xin,CNNin], axis=-1)

    for n in range(CNNlayers):
        if n == 0:
            layer = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')
            x = layer(CNNin)
            layer = Convolution2D( 32 , kernel_size= 3 , padding = 'same', activation = 'relu' )
            x = layer(x)
            layer = Dropout( .01 )
            x = layer(x)

        else:
            layer = Convolution2D( 32 , kernel_size= 3  , padding = 'same', activation = 'relu' )
            x = layer(x)
            layer = Dropout( .01 )
            x = layer(x)

    layer = Convolution2D(5 , kernel_size= 3, padding = 'same' , activation = 'relu')
    x = layer(x)
    x = Flatten()(x)

    x = concatenate([x_f1 ,x], axis=-1)

    for n in range(Denselayers):
        if n == 0 :
            """        layer1 = Dense( int(Denseoutdim/2) , name = 'Denser_'+str(n) , activation='relu')
            layer2 = Dense( int(Denseoutdim/2) , name = 'Densei_'+str(n) , activation='relu')
            #x = layer(x)
            x_r = layer1(x)
            layer = Dropout( .01 )
            x_r = layer(x_r)

            x_i = layer2(x)
            layer = Dropout( .01 )
            x_i = layer(x_i)
            """
            layer = Dense( Denseoutdim , name = 'Denser_'+str(n) , activation='relu')
            x = layer(x)
        else:
            """        layer = Dense(Denseoutdim ,  name = 'Dense_'+str(n) ,  activation='relu')
            nx_i = neg(x_i)
            x_1 = concatenate([x_r, nx_i], axis=-1)
            x_2 = concatenate([x_i, x_r], axis=-1)

            x_r = layer(x_1)
            layer = Dropout( .01 )
            x_r = layer(x_r)

            x_i = layer(x_2)
            layer = Dropout( .01 )
            x_i = layer(x_i)
            """
            layer = Dense( Denseoutdim , name = 'Denser_'+str(n) , activation='relu')
            x = layer(x)

    """    nx_i = neg(x_i)
        x_1 = concatenate([x_r, nx_i], axis=-1)
        x_2 = concatenate([x_i, x_r], axis=-1)
    """
    layer = Dense( YTRAIN.shape[1]*YTRAIN.shape[2] , name = 'finalDense'+str(n) , activation='relu')
    x = layer(x)
    #x_r = layer(x_1)
    #x_i = layer(x_2)

    #x_r =  Reshape( (256,256) )(x_r)
    #x_i =  Reshape( (256,256) )(x_i)
    x =  Reshape( (256,256)  , name = 'outputfinal')(x)

    #xcomplex = Lambda(lambda x: tf.complex(x[0], x[1]))([x_r, x_i])

    #print(xcomplex)
    #layer = Lambda( lambda x : tf.real( tf.spectral.ifft2d(x)) , output_shape=(256,256)  , name = 'outputfinal')

    #layer =  Lambda(lambda x :  tf.real( x ), output_shape=(256,256) )

    #output = layer(xcomplex)

    print(output)
    with tf.device('/cpu:0'):
        model = Model(inputs = [inputrnn , CNNin ] , outputs = [ output , ximg ])
    #model = keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
    #o  = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #o  = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #o = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    o = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

else:
    model = print('Load the model..')
    modelfile = './model-09_12_2019_12_15_30_959846.h5'
    print('load1')
    model = load_model(modelfile, custom_objects= {'subtract': subtract , 'tf' : tf } )

    #o  = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #o  = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #o = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    o = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

model.compile( optimizer=o, loss='mean_squared_error', metrics=['mae'])

#o = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#o = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#o = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#o  = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
max_len = 3000
mc = ModelCheckpoint(modelfile, monitor = 'val_outputfinal_mean_absolute_error', mode = 'min', verbose = 1, save_best_only = True)
es = EarlyStopping(monitor = 'val_outputfinal_mean_absolute_error', mode = 'min', verbose = 2, patience = es_patience)


print('')
print('Train the model...')
print( 'xrows')
print(XTRAINrows.shape)
print('xchannels')
print(XTRAINchannels.shape)

history = model.fit( { 'Seqin': XTRAINrows ,'ChannelsIn':XTRAINchannels} , [YTRAIN,YTRAIN] , batch_size = None , verbose = 2 , epochs = 1000 ,  validation_data=([XVALIDrows, XVALIDchannels], [YVALID,YVALID] ), callbacks=[es, mc])
