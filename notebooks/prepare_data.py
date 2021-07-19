#imports--------------------------------------------------------------------------------------------------------------------
import os
import glob
import sys
#import wget
import time
import subprocess
import shlex
import sys
import warnings
import random
import pickle

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

#class definitions--------------------------------------------------------------------------------------------------------------------
#scaler
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
        self._scaler = IncrementalPCA(copy = True, **kwargs)
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

#global parameters--------------------------------------------------------------------------------------------------------------------
verbose = True
#how many components to keep after PCA?
components = 300
#clipping value for FFT components (how many components should be kept?)
maxFFTComponents = 100
#amount of properties stored in the voxels
propAmount = 12
#amino acids supported by protfeat
legalAANames = {b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'G', b'H', b'I', b'L', b'K', b'M', b'F', b'P', b'S', b'T', b'W', b'Y', b'V', b'B', b'Z', b'X'}
#working on a sample? how big?
sampling = True
sampleSize = 20
#function definitions--------------------------------------------------------------------------------------------------------------------

#fit the components of the output space
#y: array of stacked distmats (on the 1st axis)
def fit_y( y, components = components, FFT = False):
    if FFT == True:
        #got through a stack of structural distmats. these should be 0 padded to all fit in an array
        
        y = np.stack([ np.fft.rfft2(y[i,:,:]) for i in range(y.shape[0])] )
        if verbose:
            print(y.shape)
        y =  np.hstack( [ np.real(y) , np.imag(y)]  )
    if verbose:
        print(y.shape)
    ndpca = NDSPCA(n_components=components)
       
    ndpca.fit(y)
    if verbose:
        print('explained variance')
        print(np.sum(ndpca.explained_variance_ratio_))
    
    scaler0 = NDSRobust()
    scaler0.fit(y)

    return ndpca, scaler0

def transform_y(y, scaler0, ndpca, FFT = False):
    if FFT == True:
        y = np.stack([np.fft.rfft2(y[i,:,:]) for i in range(y.shape[0])])
        if verbose:
            print(y.shape)
        y =  np.hstack( [ np.real(y) , np.imag(y)]  )
    
    y = ndpca.transform(y)
    scaler0 = NDSRobust()
    scaler0.fit(y)
    y = scaler0.transform(y)
    if verbose:
        print(y.shape)
    
    return y, scaler0

def inverse_transform_y(y, scaler0, ndpca, FFT=False):
    y = scaler0.inverse_transform(y)
    y = ndpca.inverse_transform(y)
    
    if FFT == True:
        split = int(y.shape[1]/2)
        y = np.stack([ np.fft.irfft2(y[i,:split,:] + 1j*y[i,split:,:]) for i in range(y.shape[0]) ] )
        
    return y

#fit the components of the in space
#stacked align voxels (on the 1st axis)
def fit_x(x, components = components, FFT = False):
    if FFT == True:
        #got through a stack of align voxels. these should be 0 padded to all fit in an array
        
        x = np.stack([ np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])] )
        if verbose:
            print(x.shape)
        x =  np.hstack( [ np.real(x) , np.imag(x)]  )
    if verbose:
        print(x.shape)
    ndpca = NDSPCA(n_components=components)
    
    ndpca.fit(x)
    if verbose:
        print('explained variance')
        print(np.sum(ndpca.explained_variance_ratio_))
    
    scaler0 = NDSRobust()
    scaler0.fit(x)
    
    return ndpca, scaler0

def transform_x(x, scaler0, ndpca, FFT = False):
    if FFT == True:
        x = np.stack([ np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])] )
        if verbose:
            print(x.shape)
        x =  np.hstack( [ np.real(x) , np.imag(x)]  )

    x = ndpca.transform(x)
    scaler0 = NDSRobust()
    scaler0.fit(x)
    x = scaler0.transform(x)
    if verbose:
        print(x.shape)
    
    return x, scaler0

def inverse_transform_x(x, scaler0, ndpca, FFT=False):
    x = scaler0.inverse_transform(x)
    x = ndpca.inverse_transform(x)
    
    if FFT == True:
        split = int(x.shape[1]/2)
        x = np.stack([ np.fft.irfftn(x[i,:split,:,:] + 1j*x[i,split:,:,:]) for i in range(x.shape[0]) ] )
        
    return x

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

#structs is a dictionary of locations of the files for structures
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

def generateProtFeatDict(sequence):
    features = FeatureGen.Get_Protein_Feat(sequence)
    return features

#generate complete set of dictionary keys generated by protFET
def protFeatKeys(align_array):
    dictKeys = set()

    for i in range(align_array.shape[0]):
        sequence = alnArrayLineToSequence(align_array, i)
        #sequence = str(msa[i].seq)
        #temporary fix for ProtFeat not supporting B, Z, X
        sequence = sequence.replace('B', 'D')
        sequence = sequence.replace('Z', 'E')
        sequence = sequence.replace('X', 'A')
        sequence = sequence.replace('.', '')
        sequence = sequence.replace('-','')
        dictKeys = dictKeys.union(set(generateProtFeatDict(sequence).keys()) - dictKeys)
    
    return dictKeys
    
#generate ProtFET array for given align (maxKeys: all keys of the feature dictionary, over the entire set)
def alignToProtFeat(align_array, dictKeys):
    #generate 2d array of ProtFET features for each sequence in align
    align_features = np.zeros((align_array.shape[0], len(dictKeys)), dtype=float)
    missingFeatures = set()

    for i in range(align_array.shape[0]):
        sequence = alnArrayLineToSequence(align_array, i)
        #temporary fix for ProtFeat not supporting B, Z, X
        sequence = sequence.replace('B', 'D')
        sequence = sequence.replace('Z', 'E')
        sequence = sequence.replace('X', 'A')
        sequence = sequence.replace('.', '')
        sequence = sequence.replace('-','')
        featuresDict = generateProtFeatDict(sequence)
        missingFeatures = dictKeys - set(featuresDict.keys())
        for newKey in missingFeatures:
            featuresDict[newKey] = float(0)
        features = np.array(list(featuresDict.values()))
        align_features[i,:] = features
        
    return align_features

#generate array of ProtFeat features for all aligns
def protFeatArrays(aligns):
    maxKeys = set()
    mostProts = biggestAlignShape(aligns)[0]
    
    #build set of all keys used in the set
    for i in range(len(aligns)):
        maxKeys = maxKeys.union(protFeatKeys(aligns[i]) - maxKeys)
           
    setFeatures = np.zeros((len(aligns), mostProts, len(maxKeys)))
    for i in range(len(aligns)):
        np.append(setFeatures, alignToProtFeat(aligns[i], maxKeys))
    
    return setFeatures

def generateGapMatrix(align_array):
    gap_array = np.array([[1 if (align_array[i][j] == b'.' or align_array[i][j] == b'-') else 0 for j in range(align_array.shape[1])] for i in range(align_array.shape[0])])
    
    return gap_array

def generateAlignVoxel(align_array, propAmount = 12, verbose = False):
    align_prop_voxel = np.zeros((align_array.shape[0], align_array.shape[1], propAmount + 1), dtype=float)
    if(verbose):
        print('final voxel shape: ', align_prop_voxel.shape)
    gap_array = generateGapMatrix(align_array)
    
    if(verbose):
        print('initial array shape: ', align_array.shape)
    
    for prop in numerical:
        align_prop_array = np.zeros(align_array.shape, dtype=float)
        align_prop_array = [[properties[prop][bstring] for bstring in seq] for seq in align_array]
        align_prop_voxel[:,:,numerical.index(prop)] = align_prop_array
    align_prop_voxel[:,:,12] = gap_array
        
    if(verbose):
        print('full voxel shape: ', align_prop_voxel.shape)

    return align_prop_voxel

#fourier transform of all aligns, input is list of unpadded aligns, output is list of FFT of aligns
def fourierAligns(aligns, verbose = False):
    alignsFFT = []
    
    for align in aligns:
        if(verbose):
            print('pre-FFT: ', align.shape)
        temp = np.fft.rfftn(align)
        if(verbose):
            print('post-FFT: ', temp.shape)
        temp = np.dstack([np.real(temp), np.imag(temp)])
        if(verbose):
            print('post-stack: ', temp.shape)
        alignsFFT.append(temp)
        
    return alignsFFT

def fourierAlign(align):
    temp = np.fft.rfftn(align)
    alignFFT = np.dstack([np.real(temp), np.imag(temp)])

    return alignFFT

def clipAlign(align):
    final = np.zeros((clippingSize, clippingSize, propAmount + 2)) #for some reason we gain 1 depth layer after FFT, so it's +2 and not +1
    if (align.shape[0] <= clippingSize and align.shape[1] <= clippingSize):
        final[:align.shape[0],:align.shape[1],:align.shape[2]] = align
    elif (align.shape[0] <= clippingSize and align.shape[1] > clippingSize):
        final[:align.shape[0],:,:align.shape[2]] = align[:,:clippingSize,:]
    elif (align.shape[0] > clippingSize and align.shape[1] <= clippingSize):
        final[:,:align.shape[1],:align.shape[2]] = align[:clippingSize,:,:]
    else:
        final[:,:,:align.shape[2]] = align[:clippingSize,:clippingSize,:]

    return final

#generate 4D array of stacked 3D voxels for PCA
def generateVoxelArray(aligns, propAmount = 12, clippingSize = maxFFTComponents, verbose = False):
    #generate voxel array
    alignsList = []
    for i in range(len(aligns)):
        alignsList.append(generateAlignVoxel(aligns[i], propAmount, verbose))
    
    #apply fourier transform to aligns before padding
    alignsList = fourierAligns(alignsList, verbose)
    
    #pad or clip all aligns to be the same size, based on how many components of the FFT we want to keep
    for i in range(len(alignsList)):
        final = np.zeros((clippingSize, clippingSize, propAmount + 2)) #for some reason we gain 1 depth layer after FFT, so it's +2 and not +1
        if(alignsList[i].shape[0] <= clippingSize and alignsList[i].shape[1] <= clippingSize):
            final[:alignsList[i].shape[0],:alignsList[i].shape[1],:alignsList[i].shape[2]] = alignsList[i]
        elif(alignsList[i].shape[0] <= clippingSize and alignsList[i].shape[1] > clippingSize):
            final[:alignsList[i].shape[0],:,:alignsList[i].shape[2]] = alignsList[i][:,:clippingSize,:]
        elif(alignsList[i].shape[0] > clippingSize and alignsList[i].shape[1] <= clippingSize):
            final[:,:alignsList[i].shape[1],:alignsList[i].shape[2]] = alignsList[i][:clippingSize,:,:]
        else:
            final[:,:,:alignsList[i].shape[2]] = alignsList[i][:clippingSize,:clippingSize,:]
        alignsList[i] = final
    
    voxels = np.stack(alignsList, axis=0)
    if verbose:
        print('voxels shape: ', voxels.shape)
    
    return voxels

#keep only chains with usable data (between 50 and 1500 AAs long, corresponding to a pfam MSA), returns list of pdb_id-chain tuples meeting requirements (pass this list to filterDataFrameBefore to remove all non-usable chains)
def filterChains(structs, availableChainData):
    validChainsList = list()
    for s in structs:
        Structure = PDBParser().get_structure(s, structs[s])
        for model in Structure:
            for chain in model:
                chainLetter = ''.join([c for c in str(chain) if c.isupper()])[1:]
                if(len(chain) < 50 or len(chain) > 1500):
                    continue
                elif chainLetter not in set(availableChainData[availableChainData['PDB'] == s]['CHAIN'].tolist()):  #checking if the chain has corresponding pfam data 
                    continue
                else:
                    validChainsList.append((s, chainLetter))
                    
    return validChainsList

def filterDataFrameBefore(validChainsList, data_df):
    keep_indexes = list()
    for i in list(data_df.index.values):
        if (data_df.loc[i, 'PDB'], data_df.loc[i, 'CHAIN']) in validChainsList:
            keep_indexes.append(i)
        
    data_df = data_df[data_df.index.isin(keep_indexes)]
    
    return data_df

#after filtering the distmat data, the dataframe must be adjusted to only include valid chain-pfam couplings and to excluse empty chains
def filterDataFrameAfter(data_df, proteinList, protChainIndexes, verbose = False):
    '''multiple pfam files are sometimes used to represent the same chain, for now only the first is used
       in the future, restructuring the data prep code could allow to keep all pfam data'''
    proteinChainLetters = list()
    proteinRepList = list()

    for protein in proteinList:
        for chain in protChainIndexes[protein].keys():
            proteinRepList.append(protein)
            proteinChainLetters.append(''.join([c for c in str(chain) if c.isupper()])[1:])

    chainLettersTuples = list(zip(proteinRepList, proteinChainLetters))

    keep_indexes = list()
    no_dupes = list()
    for i in list(data_df.index.values):
        if (data_df.loc[i, 'PDB'], data_df.loc[i, 'CHAIN']) in chainLettersTuples:
            if (data_df.loc[i, 'PDB'], data_df.loc[i, 'CHAIN']) not in no_dupes:
                no_dupes.append((data_df.loc[i, 'PDB'], data_df.loc[i, 'CHAIN']))
                keep_indexes.append(i)

    data_df = data_df[data_df.index.isin(keep_indexes)]
    if verbose:
        print(data_df)

    return data_df

#builds a dictionary of distmats in the set - structs is a dictionary of all the structures (which are then subdivided into chains)
#also adds the distmats to the corresponding data_df column
def PDBToDistmat(structs, data_df, keepOnlyFirstChain, verbose = False):
    distances = {}
    for s in structs:
        Structure = PDBParser().get_structure(s, structs[s])
        if(verbose):
            print(Structure)
        distances[s] = {}
        for model in Structure:
            for chain in model:
                if(verbose):
                    print('chain: ', chain)
                    print(len(chain))
                res = [r for r in chain.get_residues()]
                distmat = [ [res2['CA'] - res1['CA'] if 'CA' in res1 and 'CA' in res2 and i > j else 0 for i,res1 in enumerate(res)] for j,res2 in enumerate(res)]
                distmat = np.array(distmat)
                distmat+= distmat.T
                distances[s][chain] = distmat
                chainLetter = ''.join([c for c in str(chain) if c.isupper()])[1:]

                sliced_data_df = data_df.loc[(data_df['PDB'] == s) & (data_df['CHAIN'] == chainLetter)]
                if not sliced_data_df.empty:
                    distmatList = list()
                    for i in range(data_df.loc[(data_df['PDB'] == s) & (data_df['CHAIN'] == chainLetter)].shape[0]):
                        distmatList.append(distmat)
                    sliced_data_df['DISTMAT'] = distmatList
                    data_df.loc[(data_df['PDB'] == s) & (data_df['CHAIN'] == chainLetter)] = sliced_data_df
                    
                #the following condition on chain size is arbitrary and seems to work for now. chains too short or long were ignored for some reason (get_residues?). numbered chains may or may not be ok to use
                '''or str(chain) == '<Chain id=0>' or str(chain) == '<Chain id=1>' or str(chain) == '<Chain id=2>' or str(chain) == '<Chain id=3>' or str(chain) == '<Chain id=4>' or str(chain) == '<Chain id=5>' or str(chain) == '<Chain id=6>' or str(chain) == '<Chain id=7>' or str(chain) == '<Chain id=8>' or str(chain) == '<Chain id=9>' '''
                if(len(chain) < 50 or len(chain) > 1500):
                    if(verbose):
                        print('continuing')
                    continue
    
    return distances, data_df

#fourier transform of all distmats, input is list of unpadded distmats, output is list of FFT of distmats
def fourierDistmats(distmats):
    distmatsFFT = []
    
    for distmat in distmats:
        temp = np.fft.rfftn(distmat)
        temp = np.hstack([np.real(temp), np.imag(temp)])
        distmatsFFT.append(temp)
        
    return distmatsFFT

def fourierDistmat(distmat):
    temp = np.fft.rfftn(distmat)
    distmatFFT = np.hstack([np.real(temp), np.imag(temp)])

    return distmatFFT

def inverseFourierDistmats(distmatsFFT, verbose = False):
    restored_distmats = []
    
    for distmat in distmatsFFT:
        split = int(distmat.shape[1]/2)
        temp = np.fft.irfft2(distmat[:split,:] + 1j*distmat[split:,:])
        restored_distmats.append(temp)
                        
    return restored_distmats

def inverseFourierDistmat(distmatFFT, verbose = False):
    split = int(distmatFFT.shape[1]/2)
    restored_distmat = np.fft.irfft2(distmatFFT[:split,:] + 1j*distmatFFT[split:,:])
                        
    return restored_distmat

def clipDistmat(distmat):
    final = np.zeros((clippingSize, clippingSize))
    if(distmat.shape[0] <= clippingSize and distmat.shape[1] <= clippingSize):
        final[:distmat.shape[0], :distmat.shape[1]] = distmat
    elif(distmat.shape[0] <= clippingSize and distmat.shape[1] > clippingSize):
        final[:distmat.shape[0], :] = distmat[:,:clippingSize]
    elif(distmat.shape[0] > clippingSize and distmat.shape[1] <= clippingSize):
        final[:, :distmat.shape[1]] = distmat[:clippingSize,:]
    else:
        final = distmat[:clippingSize,:clippingSize]
        if(final.shape != (clippingSize, clippingSize)):
            print('error: couldn\'t pad', distmat.shape, 'to ', final.shape)
    
    return final

#generate the dict which is used when filtering dataframe for correct chain data. distances is the entire used dataset of distmat dicts
def buildProtChainDict(distances, availableChainData, proteinsList, verbose = False):
    chainIndexDict = dict()
    protChainsIndexList = list()
    
    for protein in proteinsList:
        chainIndexDict.setdefault(protein, dict())
    
    for protein in proteinsList:
        for chain in distances[protein]:
            distArray = np.array(distances[protein][chain])
            if np.sum(distArray) != 0:   #if we leave empty chains, the pca's variance calculations don't work (division by 0)
                if str(protein)+'_'+str(chain) not in protChainsIndexList:   #checking for duplicates
                    protChainsIndexList.append(str(protein)+'_'+str(chain))
                else:
                    if(verbose):
                        print('skipping duplicate chain at ', protein, ' ', chain)
            else:
                if(verbose):
                    print('skipping empty chain at ', protein, ' ', chain)

    for protein in proteinsList:
        tempIndexDict = dict()
        for chain in protChainsIndexList:
            if chain.split('_')[0] == str(protein):
                tempIndexDict[chain.split('_')[1]] = protChainsIndexList.index(chain)
        chainIndexDict[protein] = tempIndexDict
    
    return chainIndexDict

#builds 3D array of all distmats in the set, returns dictionary of chains for each protein and their index in the 3D array
def distmatDictToArray(distances, availableChainData, proteinsList, clippingSize = maxFFTComponents, verbose = False):
    #make list of proteins, containing list of distance arrays for each chain
    protChainsList = list()
    chainDistArrayList = list()
    
    chainIndexDict = dict()
    protChainsIndexList = list()
    
    for protein in proteinsList:
        chainIndexDict.setdefault(protein, dict())
    
    for protein in proteinsList:
        for chain in distances[protein]:
            distArray = np.array(distances[protein][chain])
            chainLetter = ''.join([c for c in str(chain) if c.isupper()])[1:]
            if chainLetter in set(availableChainData[availableChainData['PDB'] == protein]['CHAIN'].tolist()):  #checking if the chain has corresponding pfam data 
                if np.sum(distArray) != 0:   #if we leave empty chains, the pca's variance calculations don't work (division by 0)
                    if str(protein)+'_'+str(chain) not in protChainsIndexList:   #checking for duplicates
                        chainDistArrayList.append(distArray)
                        protChainsIndexList.append(str(protein)+'_'+str(chain))
                    else:
                        if(verbose):
                            print('skipping duplicate chain at ', protein, ' ', chain)
                else:
                    if(verbose):
                        print('skipping empty chain at ', protein, ' ', chain)
            else:
                if(verbose):
                    print('skipping ', protein, ' ', chain, ', no pfam data available')
                    #print('letter: ', chainLetter, 'set: ', set(availableChainData[availableChainData['PDB'] == protein]['CHAIN'].tolist()))
        protChainsList.append(chainDistArrayList)
        chainDistArrayList = list()
    
    for protein in proteinsList:
        tempIndexDict = dict()
        for chain in protChainsIndexList:
            if chain.split('_')[0] == str(protein):
                tempIndexDict[chain.split('_')[1]] = protChainsIndexList.index(chain)
        chainIndexDict[protein] = tempIndexDict
    
    if(verbose):
        print(protChainsIndexList)
        print('created nested list of protein chains')
    
    #preserve original shape before flattening (not needed for now, but might be useful later)
    chainAmounts = np.zeros(len(protChainsList), dtype=int)

    for i in range(len(protChainsList)):
        chainAmounts[i] = len(protChainsList[i])
    
    if(verbose):
        print(proteinsList)
        print('amount of chains for each protein: ', chainAmounts)
    
    #flatten 2D list into 1D list
    arrayList = list()
    [[arrayList.append(protChainsList[i][j]) for j in range(chainAmounts[i])] for i in range(len(protChainsList))]
    if(verbose):
        print('amount of arrays after flattening nested list: ', len(arrayList))
    
    #apply FFT to the distmats
    arrayListFFT = fourierDistmats(arrayList)

    #pad or clip all aligns to be the same size, based on how many components of the FFT we want to keep
    for i in range(len(arrayListFFT)):
        final = np.zeros((clippingSize, clippingSize))
        if(arrayListFFT[i].shape[0] <= clippingSize and arrayListFFT[i].shape[1] <= clippingSize):
            final[:arrayListFFT[i].shape[0], :arrayListFFT[i].shape[1]] = arrayListFFT[i]
        elif(arrayListFFT[i].shape[0] <= clippingSize and arrayListFFT[i].shape[1] > clippingSize):
            final[:arrayListFFT[i].shape[0], :] = arrayListFFT[i][:,:clippingSize]
        elif(arrayListFFT[i].shape[0] > clippingSize and arrayListFFT[i].shape[1] <= clippingSize):
            final[:, :arrayListFFT[i].shape[1]] = arrayListFFT[i][:clippingSize,:]
        else:
            final = arrayListFFT[i][:clippingSize,:clippingSize]
            if(final.shape != (clippingSize, clippingSize)):
                print('error: couldn\'t pad', arrayListFFT[i].shape, 'to ', final.shape)
                print(final)
                print(arrayListFFT[i])
        arrayListFFT[i] = final
    
    if(verbose):
        print('padded all distance arrays in the list')
    
    #make 3D array of all distmats in the set
    distmats = np.stack(arrayListFFT, axis=0)
    if(verbose):
        print('size of 3D distmats array: ', distmats.shape)
    
    return distmats, arrayList, chainIndexDict

#execution--------------------------------------------------------------------------------------------------------------------
#folder setup
folders = [ 'alns' , 'templates' , 'TensorflowModels' ]
clear = False

for path in folders:
    if not os.path.isdir(path):
        os.mkdir(path)
    if clear == True:
        files = glob.glob(path+'*.pdb')
        for f in files:
            os.remove(f)

#AA property dict
propfile = './physicalpropTable.csv'
propdf = pd.read_csv(propfile)

numerical = [ 'pKa side chain', 'pka2', 'pka3',
              'PI', 'Solubility Molal', 'MW', 'charge', 'ww hydrophob scale',
              'hydr or amine', 'aliphatic', 'aromatic', 'hydrophobicity at ph7']
properties = { prop: dict(zip(propdf['letter Code' ] , propdf[prop] ) ) for prop in numerical }
properties = { prop:{c.encode(): properties[prop][c] for c in properties[prop]} for prop in properties}

#dataframe of pdb ID to pfam ID correspodance for all cath proteins
filepath = '../data/pdb_chain_pfam.csv'
pdb_chain_pfam_df = pd.read_csv(filepath, header=1)

if verbose:
    print(pdb_chain_pfam_df)

#download the PDB files
#get pdb ids
models_all = pdb_chain_pfam_df['PDB'].tolist()
models = models_all[:]
if verbose:
    print(len(models))

dl_url = 'http://files.rcsb.org/download/'
dl_url_err = 'http://files.rcsb.org/download/'
structs = {}
already = glob.glob( './templates/ftp.ebi.ac.uk/pub/databases/pdb/data/structures/all/pdb/*.ent' )

#when all the files are already downloaded, check the file exists and is not empty
for m in models:
    structfile = './templates/ftp.ebi.ac.uk/pub/databases/pdb/data/structures/all/pdb/'+'pdb'+m.lower().strip()+'.ent'
    if structfile in already:
        if os.stat(structfile).st_size != 0:
            structs[m.strip()] = structfile
            
if verbose:
    print(len(structs))

#getting the alignments
alnNames_all = pdb_chain_pfam_df['PFAM_ID'].tolist()
alnNames = alnNames_all[:]

if verbose:
    print(len(alnNames))

filepath = 'Pfam-A.seed.h5'
keys = list()
aligns = list()
newAln = {}

aligns = list()

#put Pfam data into a dataframe  
with h5py.File(filepath, 'r') as f:
    for aln in f.keys():
        if 'PF' in aln and aln != 'MACPF':
            keys.append(aln.split('.')[0])
            newAln = f.get(aln)[:]
            aligns.append(newAln)

pfamDict = {'PFAM_ID':keys, 'aligns':aligns}
pfamDF = pd.DataFrame(pfamDict)

if verbose:
    print(pfamDF)
    
#merge sifts dataframe and pfam dataframe to remove missing alignments
pdb_chain_pfam_df = pdb_chain_pfam_df.merge(pfamDF, how='inner', on='PFAM_ID')
if verbose:
    print('post-pfam merge: ', pdb_chain_pfam_df)
    print('')
    
#merge sifts dataframe and model names to remove missing pdb files
modelFiles = list()
for f in os.listdir('templates/ftp.ebi.ac.uk/pub/databases/pdb/data/structures/all/pdb/'):
    if os.path.isfile(os.path.join('templates/ftp.ebi.ac.uk/pub/databases/pdb/data/structures/all/pdb/', f)) and '(' not in os.path.splitext(f)[0]:  #sometimes multiple copies of files --> xxxx(1).file
        modelFiles.append(os.path.splitext(f)[0].lower().replace('pdb',''))
        
#modelsNoDupes = list(dict.fromkeys(models))
pdbDF = pd.DataFrame(modelFiles)
pdbDF.columns = ['PDB']
pdb_chain_pfam_df = pdb_chain_pfam_df.merge(pdbDF, how='inner', on='PDB')
if verbose:
    print('post-merges: ', pdb_chain_pfam_df)
    
#adding empty column for the distmats
pdb_chain_pfam_df['DISTMAT'] = np.nan

if sampling:
    sampleIndexes = list()
    for i in range(sampleSize):
        sampleIndexes.append(random.randrange(pdb_chain_pfam_df.shape[0]))
        
    sampleProteins = pdb_chain_pfam_df[pdb_chain_pfam_df.index.isin(sampleIndexes)]['PDB'].tolist()
    samples_df = pdb_chain_pfam_df[pdb_chain_pfam_df['PDB'].isin(sampleProteins)]
    
    print(samples_df)
    
    #only keep pdb info for sampled Pfam IDs:
    keep_list = sampleProteins
    to_keep = set(keep_list)
    print(len(to_keep))
    current = set(structs.keys())
    to_delete = list(current - to_keep)
    samples_structs = structs.copy()

    for struct in to_delete:
        samples_structs.pop(struct, None)

    print(len(samples_structs))
    #preserves the order the structs should be saved in
    print(keep_list)

    structs_data = samples_structs
    data_df = samples_df
else:
    structs_data = structs
    data_df = pdb_chain_pfam_df

columnList = {'PDB', 'CHAIN'}
availableChainData = data_df[columnList]

if verbose:
    print('parsing PDB files...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    structseqs = parsePDB(structs_data)

if verbose:
    print('building distmat dictionaries...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    distances = PDBToDistmat(structs_data, verbose)

protChainIndexes = dict()

if verbose:
    print('building distmat arrays...')

#build final protein list by removing dupes but keeping order
proteinList = list()
proteinListDupes = data_df['PDB'].tolist()
proteinsAlready = set()
for p in proteinListDupes:
    if p in proteinsAlready: continue
    proteinList.append(p)
    proteinsAlready.add(p)

distmats, arrayListForTest, protChainIndexes = distmatDictToArray(distances, availableChainData, proteinList, maxFFTComponents, verbose)
if verbose:
    print('distmats shape: ', distmats.shape)

#after filtering the distmat data, the dataframe must be adjusted
data_df = filterDataFrame(data_df, proteinList, protChainIndexes, verbose)
aligns = list(data_df['aligns'])


#temporary fix: ProtFeat cannot handle B (asparagine or aspartate) in sequence, replacing all Bs with aspartate (D)
#ProtFeat cannot handle Z (Glutamate or Glutamine) in sequence, replacing all Zs with glutamate (E)
#ProtFeat cannot handle X (unknown AA) in sequence, replacing all Xs with alanine (A) -- arbitrarily
#also added fix to the ProtFeat functions to replace B, Z, X in the sequence string
if verbose:
    print('replacing ambiguous letters in the alignments...')
i = 0
j = 0
k = 0

for align in aligns:
    for seq in align:
        for aa in seq:
            if aa == b'B':
                aligns[k][i][j] = b'D'
            if aa == b'Z':
                aligns[k][i][j] = b'E'
            if aa == b'X' or aa not in legalAANames:
                aligns[k][i][j] = b'A'
            j+= 1
        i += 1
        j = 0
    k += 1
    i = 0
    j = 0
    
if verbose:
    print('n° of aligns: ', len(aligns))

if verbose:
    print('building voxels...')
voxels = generateVoxelArray(aligns, propAmount, maxFFTComponents, verbose)

if verbose:
    print("VOXELS: ")
ndpcaX, scalerX = fit_x(voxels, components = 30, FFT = False)
transformed_voxels, scalerX = transform_x(voxels, scalerX, ndpcaX, FFT = False)

if verbose:
    print("DISTMATS: ")
ndpcaY, scalerY = fit_y(distmats, components = 30, FFT = False)
transformed_distmats, scalerY = transform_y(distmats, scalerY, ndpcaY, FFT = False)

np.save('voxels', voxels)
np.save('transformed_voxels', transformed_voxels)
np.save('distmats', distmats)
np.save('transformed_distmats', transformed_distmats)   
    
with open('ndpcaX.pkl', 'wb') as f:
    pickle.dump(ndpcaX, f)
with open('scalerX.pkl', 'wb') as f:
    pickle.dump(scalerX, f)
with open('ndpcaY.pkl', 'wb') as f:
    pickle.dump(ndpcaY, f)
with open('scalerY.pkl', 'wb') as f:
    pickle.dump(scalerY, f)


'''######order of operations for parallel processing######
-filterChains(structs) to obtain validChainsList --> use with warnings.catch_warnings(): warnings.simplefilter('ignore')
-filterDataFrameBefore(validChainsList, data_df)
-distances = PDBtoDistmat(structs, data_df) --> use with warnings.catch_warnings(): warnings.simplefilter('ignore'). this also adds the distmats into the corresponding dataframe column
-buildProtChainDict(distances) to obtain chainIndexDict
-filterDataFrameAfter(data_df, proteinList, chainIndexDict)

-generateAlignVoxel() (calls generateGapMatrix())
-fourierAlign()
-clipAlign()
--> voxels.h5 (use data_df index as unique name)

-fourierDistmat()
-clipDistmat()
--> distmats.h5 (use data_df index as unique name)
'''