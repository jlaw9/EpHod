"""
Data processing for training PyTorch models
"""




#=============#
# Imports
#=============#

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

#from Bio import SeqIO
import os
import sys
from pathlib import Path
from . import utils #, evals









class XYData(Dataset):
    '''Class to load and preprocess protein embeddings'''
    
    
    def __init__(self, X, y, weights, dtype=torch.float32):
        
        super(XYData, self).__init__()
                 
        assert len(X) == len(y) == len(weights)
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype) #1D array of target labels (pHopt)
        self.weights = torch.tensor(weights, dtype=dtype)
        self.len = len(y) #Number of samples in data        

         
    def __getitem__(self, index):
        
        return (self.X[index], self.y[index], self.weights[index])


    def __len__(self):

        return self.len
    
    
    
    
    
    
    
    
 
class SequenceData(Dataset):
    '''Class to load and preprocess protein sequence data'''
    
    
    def __init__(self, seqs, y, weights, maxlen=1024, flatten=False, dtype=torch.float32):
        
        super(SequenceData, self).__init__()
                 
        
        self.Xint = [utils.categorical_encode_sequence(str(seq)) for seq in seqs]
        self.seqlens = [len(seq) for seq in seqs]
        self.maxlen = maxlen
        self.y = torch.tensor(y, dtype=dtype) #1D array of target labels (pHopt)
        self.weights = torch.tensor(weights, dtype=dtype)
        self.flatten = flatten
        self.dtype = dtype
        self.len = len(y) #Number of samples in data        



        
    def __getitem__(self, index):
        
        # One hot encode sequence
        X = utils.categorical_to_one_hot(self.Xint[index], maxlen=self.maxlen,
                                         exclude_noncanonical=True)
        mask = torch.tensor(np.max(X, axis=0), dtype=self.dtype)
        if self.flatten:
            X = X.flatten()
        X = torch.tensor(X, dtype=self.dtype)
        y = self.y[index]
        weights = self.weights[index]
                        
        return (X, y, weights, mask)




    def __len__(self):

        return self.len
    
    
    
    
    
    
    
    
class EmbeddingData(Dataset):
    '''Class to load and preprocess protein embeddings'''
    
    
    def __init__(self, 
                 accessions, 
                 y, 
                 weights=None, 
                 embedding_dir='./', 
                 tmp_dir=None, 
                 use_mask=True, 
                 maxlen=1024, 
                 dtype=torch.float32
                 ):
        
        super(EmbeddingData, self).__init__()
                 
        assert len(accessions) == len(y) == len(weights)
        assert os.path.exists(embedding_dir)
      
        self.accessions = accessions #1D array of accession codes of sequences
        self.y = torch.tensor(y, dtype=dtype) #1D array of target labels (pHopt)
        self.len = len(y) #Number of samples in data
        if weights is None: # 1D array of sample weights for reweighting
            weights = np.ones((self.len))
        else:
            self.weights = torch.tensor(weights, dtype=dtype) 
        self.embedding_dir = embedding_dir #Directory conataining ESM-1b embeddings
        self.maxlen = maxlen #Length to which embeddings should be padded
        self.use_mask = use_mask #If True, return an array of ones and zeros indicating padding positions
        # A bottleneck appears to be file I/O in reading the embeddings from file
        # Give the path to the local scratch to which files will be copied
        self.tmp_dir = tmp_dir
        if tmp_dir is not None:
            print(f"Copying embedding files locally to {tmp_dir}")
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        
        

         
    # TODO update to build the embeddings on-the-fly(?)
    def __getitem__(self, index):
        
        # Get embedding from disk, shape is [1280, seqlen]
        full_file_path = f"{self.embedding_dir}/{self.accessions[index].split('_')[0]}.pt"
        if self.tmp_dir is None:
            X = torch.load(full_file_path) 
        else:
            tmp_file_path = f"{self.tmp_dir}/{self.accessions[index].split('_')[0]}.pt"
            if os.path.isfile(tmp_file_path):
                X = torch.load(tmp_file_path) 
            else:
                # copy the file to local storage to hopefully speed-up reading the embeddings
                X = torch.load(full_file_path) 
                torch.save(X, tmp_file_path)
        X = torch.Tensor(X.T)
        seqlen = X.shape[-1]
        
        # Post-pad sequence embedding with zeros to maxlen
        padsize = self.maxlen - seqlen
        X = X[None,:,:]                 # Shape is [1, 1280, seqlen]
        X = F.pad(X, pad=(0, padsize))  # Shape is [1, 1280, maxlen]
        X = X[0]                    # Shape is [1280, maxlen]
        
        # Padding data for masking
        if self.use_mask:
            mask = ([1] * seqlen) + ([0] * padsize) # 1 for non-padded, 0 for padded positions
        else:
            mask = ([1] * self.maxlen) # Ones at all positions, no masking
        mask = torch.tensor(mask, dtype=torch.int32) # Shape is [maxlen]
        
        return (X, self.y[index], self.weights[index], mask)




    def __len__(self):

        return self.len
    








if __name__ == '__main__':
    heads, seqs = utils.read_fasta('data/preprocessing/brenda/clusters/validation_sequences.fasta')
    y = np.ones(len(seqs))
    dataset = SequenceData(seqs, y=y, weights=y, flatten=True)
    np.sum(dataset[0][3].numpy())  # Check mask






        
