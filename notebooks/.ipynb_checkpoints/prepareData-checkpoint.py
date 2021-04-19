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

sys.path.append('./ProFET/ProFET/feat_extract/')
import FeatureGen

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import h5py

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

#fit the components of the output space
#stacked distmats (on the 1st axis)
def fit_y( y , components = 300 , FFT = True ):
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

#fit the components of the in space
#stacked align voxels (on the 1st axis)
def fit_x(x, components = 300, FFT = True):
    if FFT == True:
        #got through a stack of align voxels. these should be 0 padded to all fit in an array
        
        x = np.stack([ np.fft.rfftn(x[i,:,:,:]) for i in range(x.shape[0])] )
        print(x.shape)
        x =  np.hstack( [ np.real(x) , np.imag(x)]  )
    print(x.shape)
    ndpca = NDSPCA(n_components=components)
    ndpca.fit(x)
    print('explained variance')
    print(np.sum(ndpca.explained_variance_ratio_))
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

def rundssp( infile , runIdentifier, path = 'dssp' , outdir='./', args = '' , verbose = False):
    if verbose == True:
        print( infile , runIdentifier , path , outdir )
    #i usually use filenames that reflect what the pipeline has done until that step
    outfile= outdir+runIdentifier+infile+".dssp"
    
    #here we write the command as a string using all the args
    args = path + ' -i '+  infile  +' -o '+ outfile + ' ' +args
    args = shlex.split(args)
    if verbose == True:
        print(args)
    p = subprocess.Popen(args)
    #return the opened process and the file it's creating
    
    #we can also use the communicate function later to grad stdout if we need to
    return p , outfile

def dssp2pandas(dsspstr):
    #read the dssp file format into a pandas dataframe
    start = False
    lines = {}
    count = 0
    for l in dsspstr.split('\n'):
        if '#' in l:
            start = True
        if start == True:
            if count > 0:
                lines[count] = dict(zip(header,l.split()))
            else:
                header = l.split()
            count +=1
    df = pd.DataFrame.from_dict( lines , orient = 'index')
    return df

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

def generateAlignVoxel(align_array, propAmount = 12):
    align_prop_array = np.zeros((align_array.shape[0], align_array.shape[1], propAmount + 1), dtype=float)
    gap_array = generateGapMatrix(align_array)
    
    for i in range(align_array.shape[0]):
        align_prop_array[i,:,:12] = [[properties[prop][bstring] for prop in numerical] for bstring in align_array[i]]
        align_prop_array[i,:,12] = gap_array[i,:]
    
    return align_prop_array

#generate 4D array of stacked 3D voxels for FFT (and PCA)
def generateVoxelArray(aligns, propAmount = 12):
    #find biggest align_array (the depth of the voxel is fixed by the number of properties)
    mostProts, longestProts = biggestAlignShape(aligns)

    #pad all aligns (with 'b'.) to be the same size
    for i in range(len(aligns)):
        padded = np.full((mostProts, longestProts), b'.')
        padded[:aligns[i].shape[0],:aligns[i].shape[1]] = aligns[i]
        aligns[i] = padded

    #generate voxel array
    voxels = np.zeros((len(aligns), mostProts, longestProts, propAmount + 1))
    for i in range(len(aligns)):
        voxels[i, :, :, :] = generateAlignVoxel(aligns[i])
    
    return voxels

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
#print(already)

#pull complexes
#for m in models:
    #structfile = './templates/'+m.upper().strip()+'.pdb'
    #if structfile not in already:
        #print(m)
    #    time.sleep(1)
    #    try:
    #        wget.download(url = dl_url + m.strip() +'.pdb' , out =structfile)
    #        structs[m] = structfile
    #    except:
    #        try:
    #            wget.download(url = dl_url + m.strip() +'.pdb' , out =structfile)
    #            structs[m] = structfile
    #        except:
    #            print('err', m )
    #    already.append(structfile)
    #else:
    #structs[m.strip()] = structfile

#when all the files are already downloaded, check the file exists and is not empty
for m in models:
    structfile = './templates/'+m.upper().strip()+'.pdb'
    if structfile in already:
        if os.stat(structfile).st_size != 0:
            structs[m.strip()] = structfile
        elif os.stat(structfile).st_size == 0:
            os.remove(structfile)

#runclustalo( infile , runIdentifier, path = 'clustalo' , outdir='./', args = '' , verbose = False)

alnNames_all = pdb_chain_pfam_df['PFAM_ID'].tolist()
alnNames = alnNames_all[:]

#print(alnNames)

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

#temporary fix: ProtFeat cannot handle B (asparagine or aspartate) in sequence, replacing all Bs with aspartate (D)
#ProtFeat cannot handle Z (Glutamate or Glutamine) in sequence, replacing all Zs with glutamate (E)
#ProtFeat cannot handle X (unknown AA) in sequence, replacing all Xs with alanine (A) -- arbitrarily
print('replacing ambiguous letters in the alignments...')
i = 0
j = 0
k = 0

for align in pfamDF['aligns']:
    temp_align = align
    for seq in align:
        for aa in seq:
            if aa == b'B':
                temp_align[i][j] = b'D'
            if aa == b'Z':
                temp_align[i][j] = b'E'
            if aa == b'X':
                temp_align[i][j] = b'A'
            j+= 1
        i += 1
        j = 0
    pfamDF['aligns'].iloc[k] = temp_align
    k += 1
    i = 0
    j = 0

#print(pfamDF)

checkMerges = True

if checkMerges:
    print('pre-merges: ', pdb_chain_pfam_df)
    print('')
    
#merge sifts dataframe and pfam dataframe to remove missing alignments
pdb_chain_pfam_df = pdb_chain_pfam_df.merge(pfamDF, how='inner', on='PFAM_ID')
if checkMerges:
    print('post-pfam merge: ', pdb_chain_pfam_df)
    print('')
    
#merge sifts dataframe and model names to remove missing pdb files
modelFiles = list()
for f in os.listdir('templates'):
    if os.path.isfile(os.path.join('templates', f)) and '(' not in os.path.splitext(f)[0]:  #sometimes multiple copies of files --> xxxx(1).file
        modelFiles.append(os.path.splitext(f)[0].lower())
        
#modelsNoDupes = list(dict.fromkeys(models))
pdbDF = pd.DataFrame(modelFiles)
pdbDF.columns = ['PDB']
pdb_chain_pfam_df = pdb_chain_pfam_df.merge(pdbDF, how='inner', on='PDB')
if checkMerges:
    print('post-merges: ', pdb_chain_pfam_df)
    
#sample 1000 alignments for PCA (then apply same scaler to everything else)
sampling = True

if sampling:
    sampleIndexes = list()
    for i in range(1000):
        sampleIndexes.append(random.randrange(pfamDF.shape[0]))
        
    samples_df = pfamDF[pfamDF.index.isin(sampleIndexes)]

    print(samples_df)
    aligns = list(samples_df['aligns'])

print('parsing PDB files...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    structseqs = parsePDB(structs)

#where do we use the dssp dataframes?
#TODO - make into functions
#for s in structs:
    #print(structs[s])
    #p, outdssp = rundssp( structs[s] , 'test' , outdir = './templates/' , verbose = True)
    #p.wait()
    
#dssps= glob.glob( './templates/*.dssp')
#print(dssps)
#for dssp in dssps:
    #with open( dssp , 'r') as dsspin:
        #df = dssp2pandas( dsspin.read() )

print('building distmat dictionaries...')
distances = PDBToDistmat(structs, show = False)
    
print('n° of aligns: ', len(aligns))
    
#features = protFeatArrays(aligns)
print('building distmat arrays...')
distmats = distmatDictToArray(distances)
print('building voxels...')
voxels = generateVoxelArray(aligns, propAmount = 12)

print("VOXELS: ")
scalerX, ndpcaX = fit_x(voxels, components = 300, FFT = True)
transformed_voxels = transform_x(voxels, scalerX, ndpcaX, FFT = True)

print("DISTMATS: ")
scalerY, ndpcaY = fit_y(distmats, components = 300, FFT = True)
transformed_distmats = transform_y(distmats, scalerY, ndpcaY, FFT = True)

#clear all the h5 files and create fresh ones (need to create them because opening in r+ mode)
clear = True

if clear:
    if os.path.isfile('voxels.h5'):
        os.remove('voxels.h5')
    if os.path.isfile('transformedVoxels.h5'):
        os.remove('transformedVoxels.h5')
    if os.path.isfile('distmats.h5'):
        os.remove('distmats.h5')
    if os.path.isfile('transformedDistmats.h5'):
        os.remove('transformedDistmats.h5')
        
    open('voxels.h5', 'a').close()
    open('transformedVoxels.h5', 'a').close()
    open('distmats.h5', 'a').close()
    open('transformedDistmats.h5', 'a').close()

assert len(models) == len(aligns) and len(aligns) == len(alnNames)

with h5py.File('voxels.h5', 'r+') as hf:
    i = 0
    name = ''
    for voxel in voxels:
        name = alnNames[i] + '_voxel'
        if not hf.get(name):
            hf.create_dataset(name, data=voxel)
        i += 1
        
with h5py.File('transformedVoxels.h5', 'r+') as hf:
    #add the scaler and pca for reverse transform -- is the info contained in the str?
    hf.create_dataset('scaler', data=str(scalerX))
    hf.create_dataset('ndpca', data=str(ndpcaX))
    
    #add the transformed voxels
    i = 0
    name = ''
    for voxel in transformed_voxels:
        name = alnNames[i] + '_transformedV'
        if not hf.get(name):
            hf.create_dataset(name, data=voxel)
        i += 1
        
with h5py.File('distmats.h5', 'r+') as hf:
    i = 0
    name = ''
    for distmat in distmats:
        name = models[i] + '_distmat'
        if not hf.get(name):
            hf.create_dataset(name, data=distmat)
        i += 1
        
with h5py.File('transformedDistmats.h5', 'r+') as hf:
    #add the scaler and pca for reverse transform
    hf.create_dataset('scaler', data=str(scalerY))
    hf.create_dataset('ndpca', data=str(ndpcaY))
    
    #add the transformed distmats
    i = 0
    name = ''
    for distmat in transformed_distmats:
        name = models[i] + '_transformedD'
        if not hf.get(name):
            hf.create_dataset(name, data=distmat)
        i += 1
