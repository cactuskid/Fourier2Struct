import os
import glob
import wget
import time
import subprocess
import shlex
import sys
import warnings
import random

from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
from Bio import AlignIO

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler,  Normalizer , MinMaxScaler , RobustScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

sys.path.append('./ProFET/ProFET/feat_extract/')
import FeatureGen

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import h5py
import pickle


######GENERAL PARAMS######
sample_size = 10
batch_size = 10
components = 10 #comp amount post PCA
cutoff = 300 #amount of parameters to keep for FFT (when cutting high freqs)
propAmount = 12 #how many properties of AA there are
biggestAlnShape = {4152, 2866} #{most sequences/align, longest sequence} in pfam database
##########################


#PCA and scaler
class NDSRobust(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = RobustScaler(copy=True, **kwargs)
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

#ndimensional PCA for arrays

class NDSPCA(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = IncrementalPCA(copy = True, batch_size = batch_size, **kwargs)
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
    
#fit the components of the output space
#stacked distmats (on the 1st axis)
def fft_y(y):
    y = np.stack([np.fft.rfft2(y[i,:,:]) for i in range(y.shape[0])])
    y = np.hstack([np.real(y), np.imag(y)])
    
    return y

def fit_y( y , components = components , FFT = True ):
    if FFT == True:
        #got through a stack of structural distmats. these should be 0 padded to all fit in an array
        
        y = np.stack([ np.fft.rfft2(y[i,:,:]) for i in range(y.shape[0])] )
        print(y.shape)
        y =  np.hstack( [ np.real(y) , np.imag(y)]  )
    print(y.shape)
    ndpca = NDSPCA(n_components=components)
    ndpca.fit(y)
    print('explained variance')
    print(np.sum(ndpca.explained_variance_ratio_))
    y = ndpca.transform(y)
    scaler0 = RobustScaler( )
    scaler0.fit(y)
    return scaler0, ndpca

def transform_y(y, scaler0, ndpca, FFT = False):
    if FFT == True:
        y = np.stack([np.fft.rfft2(y[i,:,:]) for i in range(y.shape[0])])
        print(y.shape)
        y =  np.hstack( [ np.real(y) , np.imag(y)]  )
    y = ndpca.transform(y)
    print(y.shape)
    y = scaler0.transform(y)
    
    return y

def inverse_transform_y(y, scaler0, ndpca, FFT=False):
    y = scaler0.inverse_transform(y)
    y = ndpca.inverse_transform(y)
    
    if FFT == True:
        split = int(y.shape[1]/2)
        y = np.stack([ np.fft.irfft2(y[i,:split,:] + 1j*y[i,split:,:]) for i in range(y.shape[0]) ] )
        
    return y

def fft_x(x, cutoff = cutoff):
    x = np.stack([np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])])
    x = x[:,:cutoff,:,:]
    x = np.hstack([np.real(x), np.imag(x)])
    
    return x

def partial_fit_x(x, ndpca, scaler):
    ndpca = ndpca.partial_fit(x)
    
    print('explained variance')
    print(np.sum(ndpca.explained_variance_ratio_))
    print(ndpca.explained_variance_ratio_)
    #TODO - find scaler with partial fit option
    #scaler.fit(x)
    return scaler, ndpca

#fit the components of the in space
#stacked align voxels (on the 1st axis)
def fit_x(x, components = components, cutoff = cutoff, FFT = True):
    if FFT == True:
        #got through a stack of align voxels. these should be 0 padded to all fit in an array
        
        x = np.stack([ np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])] )
        x = x[:,:cutoff,:,:]
        print(x.shape)
        x =  np.hstack( [ np.real(x) , np.imag(x)]  )
    print(x.shape)
    ndpca = NDSPCA(n_components=components)
    ndpca.fit(x)
    print('explained variance')
    print(np.sum(ndpca.explained_variance_ratio_))
    print(ndpca.explained_variance_ratio_)
    x = ndpca.transform(x)
    scaler0 = RobustScaler( )
    scaler0.fit(x)
    return scaler0, ndpca

def transform_x(x, scaler0, ndpca, FFT = False):
    if FFT == True:
        x = np.stack([ np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])] )
        print(x.shape)
        x =  np.hstack( [ np.real(x) , np.imag(x)]  )
    x = ndpca.transform(x)
    print(x.shape)
    x = scaler0.transform(x)
    
    return x

#todo -- check the split is happening in the right dimension
def inverse_transform_x(x, scaler0, ndpca, FFT=False):
    x = scaler0.inverse_transform(x)
    x = ndpca.inverse_transform(x)
    
    if FFT == True:
        split = int(x.shape[1]/2)
        x = np.stack([ np.fft.irfftn(x[i,:split,:,:] + 1j*x[i,split:,:,:]) for i in range(x.shape[0]) ] )
        
    return x

#get align files
def runclustalo( infile , runIdentifier, path = 'clustalo' , outdir='./', args = '' , verbose = False):
    if verbose == True:
        print( infile , runIdentifier , path , outdir )
    #i usually use filenames that reflect what the pipeline has done until that step
    outfile= outdir+runIdentifier+infile+".aln.fasta"
    
    #here we write the command as a string using all the args
    args = path + ' -i '+  infile  +' -o '+ outfile + ' ' +args
    args = shlex.split(args)
    if verbose == True:
        print(args)
    p = subprocess.Popen(args )
    #return the opened process and the file it's creating
    
    #we can also use the communicate function later to grad stdout if we need to
    return p , outfile

#TODO - add sequence to align

def alnFileToArray(filename, returnMsa = False):
    alnfile = filename
    msa = AlignIO.read(alnfile , format = 'fasta')
    align_array = np.array([ list(rec.upper())  for rec in msa], np.character)
    
    if returnMsa:
        return align_array, msa
        
    return align_array

def alnArrayLineToSequence(align_array, index):
    seq = ''
    for aa in align_array[index]:
        seq += aa.decode('utf-8')
    
    return seq

#generate align list
def generateAlignList(directory = 'alns', returnMsa = False):
    aligns = list()
    msas = list()
    
    #read through align files to get align arrays list
    for file in os.listdir(directory):
        if file.endswith('.fasta'):
            aligns.append(alnFileToArray(directory+'/'+file, returnMsa)[0])
            if returnMsa:
                msas.append(alnFileToArray(directory+'/'+file, returnMsa)[1])
    
    if returnMsa:
        return aligns, msas
    
    return aligns

#find biggest align shape (for padding) - aligns is a list of arrays
def biggestAlignShape(aligns):
    longestProts = 0
    mostProts = 0

    for aln in aligns:
        if aln.shape[0] > mostProts:
            mostProts = aln.shape[0]
        if aln.shape[1] > longestProts:
            longestProts = aln.shape[1]
    
    return mostProts, longestProts

def generateGapMatrix(align_array):
    gap_array = np.array([[1 if (align_array[i][j] == b'.' or align_array[i][j] == b'-') else 0 for j in range(align_array.shape[1])] for i in range(align_array.shape[0])])
    
    return gap_array

def generateAlignVoxel(align_array, propAmount = propAmount):
    align_prop_array = np.zeros((align_array.shape[0], align_array.shape[1], propAmount + 1), dtype=np.float32)
    gap_array = generateGapMatrix(align_array)
    
    for i in range(align_array.shape[0]):
        align_prop_array[i,:,:propAmount] = [[properties[prop][bstring] for prop in numerical] for bstring in align_array[i]]
        align_prop_array[i,:,propAmount] = gap_array[i,:]
    
    return align_prop_array

#generate 4D array of stacked 3D voxels for FFT (and PCA)
def generateVoxelArray(aligns, propAmount = propAmount):
    #find biggest align_array (the depth of the voxel is fixed by the number of properties)
    mostProts, longestProts = biggestAlignShape(aligns)

    #pad all aligns (with 'b'.) to be the same size
    for i in range(len(aligns)):
        padded = np.full((mostProts, longestProts), b'.')
        padded[:aligns[i].shape[0],:aligns[i].shape[1]] = aligns[i]
        aligns[i] = padded

    #generate voxel array
    voxels = np.zeros((len(aligns), mostProts, longestProts, propAmount + 1), dtype=np.float32)
    for i in range(len(aligns)):
        voxels[i, :, :, :] = generateAlignVoxel(aligns[i])
    
    return voxels

folders = [ 'alns' , 'templates' , 'TensorflowModels' ]
clear = False

for path in folders:
    if not os.path.isdir(path):
        os.mkdir(path)
    if clear == True:
        files = glob.glob(path+'*.pdb')
        for f in files:
            os.remove(f)
            
propfile = './physicalpropTable.csv'
propdf = pd.read_csv(propfile)

numerical = [ 'pKa side chain', 'pka2', 'pka3',
              'PI', 'Solubility Molal', 'MW', 'charge', 'ww hydrophob scale',
              'hydr or amine', 'aliphatic', 'aromatic', 'hydrophobicity at ph7']
properties = { prop: dict(zip(propdf['letter Code' ] , propdf[prop] ) ) for prop in numerical }
properties = { prop:{c.encode(): properties[prop][c] for c in properties[prop]} for prop in properties}

#dataframe of pdb ID to pfam ID correspodance for all cath proteins
filepath = '/home/cactuskid13/struct_data/sifts/pdb_chain_pfam.csv'
pdb_chain_pfam_df = pd.read_csv(filepath, header=1)

alnNames_all = pdb_chain_pfam_df['PFAM_ID'].tolist()
alnNames = alnNames_all[:]

filepath = 'Pfam-A.full.h5'
keys = list()
aligns = list()
newAln = {}

"""done = False

with h5py.File(filepath, 'r') as f:
    for aln in alnNames:
        if aln in f.keys():
            newAln = f.get(aln)[:]
            aligns.append(newAln)
        else:
            print('error, ', aln, ' not in Pfam, removing entry')
            while not done:
                done = removeItem(aln, models, alnNames)
            done = False"""


#put Pfam data into a dataframe  
with h5py.File(filepath, 'r') as f:
    for aln in f.keys():
        if 'PF' in aln and aln != 'MACPF':
            keys.append(aln.split('.')[0])
            newAln = f.get(aln)[:]
            aligns.append(newAln)

pfamDict = {'PFAM_ID':keys, 'aligns':aligns}
pfamDF = pd.DataFrame(pfamDict)

sampleIndexes = list()
for i in range(sample_size):
    sampleIndexes.append(random.randrange(pfamDF.shape[0]))

samples_df = pfamDF[pfamDF.index.isin(sampleIndexes)]

print(samples_df)
aligns = list(samples_df['aligns'])
alnNames = list(samples_df['PFAM_ID'])

print('replacing ambiguous letters in the alignments...')

for align in aligns:
    align[align == b'B'] = b'D'
    align[align == b'U'] = b'C'
    align[align == b'Z'] = b'E'
    align[align == b'X'] = b'A'
            

print('n° of aligns: ', len(aligns))
scalerX = RobustScaler()
ndpcaX = NDSPCA(n_components = components)
batchAmount = math.ceil(len(aligns)/float(batch_size))
#linearity of the Fourier transform --> sum of fft(batch) = fft(sum of batches)
partial_ffts = list()
#TODO - padding needs to be the same for each batch --> first pass through all data to find biggest align + longest seq
voxels = generateVoxelArray(aligns, propAmount = propAmount)
print("VOXELS: ")
#TODO - iterative FFT over all data then iterative fit over all data
scalerX, ndpcaX = fit_x(voxels, components = components, FFT = True)

#writing examples of voxels for debug
with open('voxel.txt', 'w') as f:
    for i in range(5):
        f.write(str(i)+': \n')
        f.write(str(voxels[i])+'\n')

pickle.dump(scalerX, open('scalerX.pickle', 'wb'))
pickle.dump(ndpcaX, open('ndpcaX.pickle','wb'))