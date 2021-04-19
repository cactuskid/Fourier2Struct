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
batch_size = 30 #how many aligns at a time
##########################

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
            
filepath = 'Pfam-A.full.h5'
keys = list()
aligns = list()
newAln = {}

#put Pfam data into a dataframe  
with h5py.File(filepath, 'r') as f:
    for aln in f.keys():
        if 'PF' in aln and aln != 'MACPF':
            keys.append(aln.split('.')[0])
            newAln = f.get(aln)[:]
            aligns.append(newAln)

pfamDict = {'PFAM_ID':keys, 'aligns':aligns}
pfamDF = pd.DataFrame(pfamDict)

shape = {}  
shape = biggestAlignShape(aligns)

pickle.dump(shape, open('biggestAlnShape.pickle', 'wb'))