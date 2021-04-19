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
        self._scaler = IncrementalPCA(copy = True, batch_size = 50, **kwargs)
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
def fit_y( y , components = 50, FFT = True ):
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
    print(ndpca.explained_variance_ratio_)
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

#fit the components of the in space
#stacked align voxels (on the 1st axis)
def fit_x(x, components = 50, cutoff = 300, FFT = True):
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

#structs is a dictionary of all the structures (which are then subdivided into chains)
def parsePDB(structs):
    parser = PDBParser()
    converter = {'ALA': 'A', 'ASX': 'B', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
                 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P',
                 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'SEC': 'U', 'VAL': 'V', 'TRP': 'W',
                 'XAA': 'X', 'TYR': 'Y', 'GLX': 'Z'}
    structseqs={}
    with open( 'structs.fast' , 'w') as fastout:
        for s in structs:
            Structure = PDBParser().get_structure(s, structs[s])
            for model in Structure:
                for chain in model:
                    res = chain.get_residues()
                    seq =  ''.join([ converter[r.get_resname()] for r in res if r.get_resname() in converter ] )
                    fastout.write('>' + s + '|'+ chain.id +'\\n')
                    fastout.write(str( seq ) +'\\n'  )
                    structseqs[ s + '|'+ chain.id ] = seq
    
    return structseqs

#builds a dictionary of distmats in the set - structs is a dictionary of all the structures (which are then subdivided into chains)
def PDBToDistmat(structs, show = False):
    distances = {}
    for s in structs:
        Structure = PDBParser().get_structure(s, structs[s])
        distances[s] = {}
        for model in Structure:
            for chain in model:
                res = [r for r in chain.get_residues()]
                distmat = [ [res2['CA'] - res1['CA'] if 'CA' in res1 and 'CA' in res2 and i > j else 0 for i,res1 in enumerate(res)] for j,res2 in enumerate(res)]
                distmat = np.array(distmat)
                distmat+= distmat.T
                distances[s][chain] = distmat

    if show:
        for s in distances:
            print(s)
            for c in distances[s]:
                sns.heatmap(distances[s][c])
                plt.show()
    
    return distances

#builds 3D array of all distmats in the set
def distmatDictToArray(distances):
    #make list of proteins, containing list of distance arrays for each chain
    protChainsList = list()
    chainDistArrayList = list()

    for protein in distances:
        for chain in distances[protein]:
            distArray = np.array(distances[protein][chain])
            if np.sum(distArray) != 0:   #if we leave empty chains, the pca's variance calculations don't work (division by 0)
                chainDistArrayList.append(distArray)
        protChainsList.append(chainDistArrayList)
        chainDistArrayList = list()
    
    #preserve original shape before flattening (not needed for now, but might be useful later)
    chainAmounts = np.zeros(len(protChainsList), dtype=int)

    for i in range(len(protChainsList)):
        chainAmounts[i] = len(protChainsList[i])
    
    #flatten 2D list into 1D list
    arrayList = list()
    [[arrayList.append(protChainsList[i][j]) for j in range(chainAmounts[i])] for i in range(len(protChainsList))]
    
    #find size of the largest distmat
    maxX, maxY = biggestAlignShape(arrayList)

    #pad the arrays so they're all the same size
    for i in range(len(arrayList)):
        padded = np.zeros((maxX, maxY))
        padded[:arrayList[i].shape[0], :arrayList[i].shape[1]] = arrayList[i]
        arrayList[i] = padded

    #make 3D array of all distmats in the set
    distmats = np.zeros((len(arrayList), maxX, maxY))

    for i in range(len(arrayList)):
        distmats[i,:,:] = arrayList[i]
    
    return distmats

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

#get pdb ids
models_all = pdb_chain_pfam_df['PDB'].tolist()
models = models_all[:]

dl_url = 'http://files.rcsb.org/download/'
dl_url_err = 'http://files.rcsb.org/download/'
structs = {}
already = glob.glob( './templates/*.pdb' )

for m in models:
    structfile = './templates/'+m.upper().strip()+'.pdb'
    if structfile in already:
        if os.stat(structfile).st_size != 0:
            structs[m.strip()] = structfile
        elif os.stat(structfile).st_size == 0:
            os.remove(structfile)
            
modelFiles = list()
for f in os.listdir('templates'):
    if os.path.isfile(os.path.join('templates', f)) and '(' not in os.path.splitext(f)[0]:  #sometimes multiple copies of files --> xxxx(1).file
        modelFiles.append(os.path.splitext(f)[0].lower())
        
pdbDF = pd.DataFrame(modelFiles)
pdbDF.columns = ['PDB']

sampleIndexes = list()
for i in range(50):
    sampleIndexes.append(random.randrange(pdbDF.shape[0]))

samples_df = pdbDF[pdbDF.index.isin(sampleIndexes)]

print(samples_df)

samples_dict = {}
for m in list(samples_df['PDB']):
    structfile = './templates/'+m.upper()+'.pdb'
    samples_dict[m] = structfile

print('building distmat dictionaries...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    distances = PDBToDistmat(samples_dict, show = False)
    
print('building distmat arrays...')
distmats = distmatDictToArray(distances)
    
print('n° of distmas: ', len(distmats))
print("DISTMATS: ")
scalerY, ndpcaY = fit_y(distmats, components = 50, FFT = True)
#transformed_distmats = transform_y(distmats, scalerY, ndpcaY, FFT = True)

pickle.dump(scalerY, open('scalerY.pickle', 'wb'))
pickle.dump(ndpcaY, open('ndpcaY.pickle','wb'))