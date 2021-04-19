import h5py
from Bio import SeqIO, AlignIO
import io
import numpy as np
import os

pfam_path = '../Pfam-A.seed'
import gzip

def yield_alns(pfam_path , verbose = False):    
    with open(pfam_path, 'r', encoding='ISO-8859-1') as f:
        aln= ''
        acc = None
        count =0
        lCount = 0
        for i,l in enumerate( f ):
            lCount += 1
            if verbose == True:
                if i > 3000 and i < 4000:
                    print(l)
            
            if lCount < 10**6:
                aln+=l
            
            if acc is None and 'AC' in l:
                acc = l.split()[2] 
            
            if l == '//\n':
                if lCount > 10**6:
                    print(acc + 'truncated')
                if count < 8063 or count > 8065:
                    msa = AlignIO.read(io.StringIO(aln), "stockholm")
                else:
                    print('skipping')
                    msa = None
                idPfam = acc
                acc = None
                aln = ''
                lCount = 0
                yield idPfam, msa
                if count % 1000 == 0 :
                    print(msa)
                count+=1

if os.path.isfile('Pfam-A.full.h5'):
    os.remove('Pfam-A.full.h5')
open('Pfam-A.full.h5', 'a').close()

with h5py.File('Pfam-A.full' +'.h5', 'r+') as hf:
    i = 0
    for pfam_id, msa in yield_alns(pfam_path):
        i += 1
        print(i)
        if not hf.get(pfam_id):
            try:
                align_list = list()
                for rec in msa:
                    align_list.append(np.array(list(rec.upper()), np.character))
                try:
                    align_array = np.array(align_list)
                    hf.create_dataset(pfam_id,  data=align_array)
                except:
                    print('exception')
            except:
                print('for loop exception')
        else:
            print(pfam_id)
