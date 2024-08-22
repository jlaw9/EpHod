"""
Train a residual light attention top model (RLAT) with distributed training
"""






import numpy as np
import pandas as pd
import time
import argparse
from collections import OrderedDict
from tqdm import trange, tqdm
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.multiprocessing as mp

from sklearn.preprocessing import OneHotEncoder

import os
import subprocess
import sys
sys.path.insert(1, './')
sys.path.insert(1, './python_scripts/')

import warnings
warnings.filterwarnings('ignore')

#from module import utils, evals, adhoc, torchmodels, dataproc
from src import utils, dataproc

from ephod import models as torchmodels








def create_parser():
    '''Parse command-line training arguments'''
    
    parser = argparse.ArgumentParser(description="Train ResidualLightAttention")
    parser.add_argument('--trainseqs', type=str, 
                        help='File containing accession codes of training sequences in'\
                            ' csv format')
    parser.add_argument('--valseqs', type=str,
                        help='File containing accession codes of validation sequences in'\
                            ' csv format')
    parser.add_argument('--target_data', type=str,
                        help='Path to csv file containing target labels and sample '\
                            'weights')
    parser.add_argument('--embedding_dir', type=str,
                        help='Path to directory containing per-residue embeddings of'\
                             ' sequences')
    parser.add_argument('--y_cols', type=str,
                        help='Comma-separated list of columns to use as prediction'\
                             ' targets')
    parser.add_argument('--y_weights', type=str,
                        help='Comma-separated list of weights (float) to multiply to different y_cols'\
                             ' to give more weight to some targets')
    parser.add_argument('--feat_cols', type=str,
                        help='Comma-separated list of columns to use as additional'\
                             ' sequence-level features e.g., enzyme class')
    parser.add_argument('--local_storage', type=str,
                        help='Local path where embeddings will be copied to speeed-up '\
                             ' reading files')
    parser.add_argument('--model_name', type=str, 
                        help='Name of model used in saving parameters.')                             
    parser.add_argument('--paramsjson', type=str,
                        help='Json file containing hyperparameters for building and '\
                            'training ResidualLightAttention')
    parser.add_argument('--encoder', type=str,
                        help='Pickle file containing one-hot encoder for including '\
                            'extra sequence features. Will be created if not found')
    parser.add_argument('--savedir', type=str, 
                        help='Directory to save model parameters and training progress')
    parser.add_argument('--distributed', type=int,
                        help='Whether to train with a single GPU (0) or multiple GPUs(1)')
    parser.add_argument('--maxlen', default=1024, type=int,
                        help='Maximum length of sequences. Embeddings will be '\
                            'post-padded to this length with zeros')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of single GPU used in training and inference '\
                             '(i.e. local, non-distributed)')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='Number of nodes for distributed training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='The number of epochs for training')
    parser.add_argument('--reduce_lr_patience', default=20, type=int, 
                        help='Reduce learning rate by 0.5 if validation loss does not '\
                             'improve after reduce_lr_patience epochs')
    parser.add_argument('--stop_patience', default=100, type=int,
                        help='Exit training if validation loss does not improve after '\
                             'stop_patience epochs')
    parser.add_argument('--restart', default=0, type=int,
                        help='Whether to begin new training (0) and overwrite checkpoint'\
                             ' files, or to restart training (1) and append checkpoint '\
                             ' files')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print out details of training (1) or not (0)')
    args = parser.parse_args()
    if args.y_cols is not None:
        args.y_cols = [s.strip() for s in args.y_cols.split(",") if s != ""]
    if args.feat_cols is not None:
        args.feat_cols = [s.strip() for s in args.feat_cols.split(",") if s != ""]
    if args.y_weights is not None:
        args.y_weights = [float(s.strip()) for s in args.y_weights.split(",") if s != ""]
    args.params = utils.read_json(args.paramsjson)  # Model parameters as dict

    return args








def configure_cuda(args):
    '''Configure devices for training'''
    
    if not torch.cuda.is_available():
        print("FATAL: GPU not available")
        exit()
    
    torch.backends.cudnn.benchmark = True  # For efficient training
    args.gpus_per_node = torch.cuda.device_count()   
    args.num_cpus = os.cpu_count()     
    if args.distributed:
        args.world_size = args.gpus_per_node * args.num_nodes
    else:
        args.world_size = 1
    
    return args.world_size, args







def configure_processes(rank, args):
    '''Setup the processes group for distributed training'''
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    




def get_one_hot_encoder(df, feat_cols, encoder_file=None):
    if encoder_file is not None and os.path.isfile(encoder_file):
        # load the saved encoder
        with open(encoder_file, "rb") as f:
            pickle_cols, enc = pickle.load(f)
        if pickle_cols != feat_cols:
            raise ValueError(
                    f"feat_cols given do not match cols from pickle file. "
                    f"{feat_cols = }, {pickle_cols = }"
                    )
        # If the encoder already exists, then add missing feature columns as empty
        # (to handle the case where fine-tuning has extra features not present
        # in pre-training)
        for col in feat_cols:
            if col not in df.columns:
                print(f"WARNING: feat_col '{col}' not present in data. Adding empty column")
                df[col] = np.nan
                df[col] = df[col].astype(object)
    else:
        # check that the specified feature columns are present
        for col in feat_cols:
            if col not in df.columns:
                sys.stderr.write(f"ERROR: feat_col '{col}' not in df.cols: {df.columns}")
                df[col]
        # embed the features using a one-hot encoding
        enc = OneHotEncoder(drop=[np.nan] * len(feat_cols))
        enc.fit(df[feat_cols])
        # now save the encoder to a pickle file
        if encoder_file is not None:
            with open(encoder_file, "wb") as f:
                pickle.dump((feat_cols, enc), f)
    return enc, df


    
    
def get_dataset(seqs, 
                sample_method, 
                args,
                y_cols=None, 
                feat_cols=None, 
                **kwargs):
    '''Get dataset of ESM embeddings data'''

    if y_cols is None:
        y_cols = ["ph_opt", "ph_growth_range"]
    print(f"Reading seq_file: {seqs}, target_data_file: {kwargs['target_data']}")
    accessions = pd.read_csv(seqs, index_col=0).index # Seq. accession codes
    accessions = list(accessions)
    target_data = pd.read_csv(kwargs["target_data"], index_col=0)
    target_data = target_data.replace({"?": np.nan, "-": np.nan})
    print(f"{len(accessions) = }, {len(target_data) = }")
    for col in y_cols:
        if col not in target_data.columns:
            sys.stderr.write(f"ERROR: y_col '{col}' not in target_data.cols: {target_data.columns}")
            target_data[col]
    #accessions_with_file = []
    #for a in accessions:
    #    if os.path.isfile(f"{args.embedding_dir}/{a}.pt"):
    #        accessions_with_file += [a]
#
#    print(f"{len(accessions_with_file) = }")
    # limit the target data to the given accessions
    # The sequence ID is not longer unique since I'm including
    # different features per sequence (e.g., oxidation vs reduction reaction)
    target_data = target_data[target_data.index.isin(accessions)]
    accessions = target_data.index
    print(f"{len(accessions) = }, {len(target_data) = }")
    seq_features = None
    args.num_seq_features = 0
    # Encode extra sequence-level features using a one-hot encoding
    if feat_cols is not None:
        enc, target_data = get_one_hot_encoder(target_data, feat_cols, encoder_file=args.encoder) 
        #feat_data = target_data.loc[accessions,feat_cols]
        feat_data = target_data[feat_cols]
        print("Applying one-hot encoding to data (if this hangs, investigate pandas.get_dummies)")
        seq_features = enc.transform(feat_data).toarray()
        args.num_seq_features = len([c for cat in enc.categories_ for c in cat if not pd.isnull(c)]) 
        print(enc.categories_)
        print(f"{args.num_seq_features = }")

    #print(len(accessions), len(target_data.loc[accessions,y_cols].values))
    #print(len(set(accessions)), len(target_data.loc[accessions,y_cols].index), target_data.loc[accessions,y_cols].index)
    #accessions = target_data.loc[accessions,y_cols].index

    dataset = dataproc.EmbeddingData(
        accessions=accessions, 
        y=target_data[y_cols].values,
        seq_features=seq_features,
        weights=target_data[sample_method].values, 
        embedding_dir=kwargs["embedding_dir"], 
        use_mask=True, 
        maxlen=kwargs["maxlen"],
        tmp_dir=kwargs["local_storage"]
        )
    
    return dataset, args








def get_dataloader(rank, world_size, dataset, train_mode, args):
    '''Return a dataloader for distributed training/inference'''
    
    #sampler = DistributedSampler(dataset, 
    #                             num_replicas=world_size, 
    #                             rank=rank,
    #                             shuffle=train_mode, 
    #                             drop_last=train_mode)
    dataloader = DataLoader(dataset, 
                            #sampler=sampler,
                            shuffle=train_mode,
                            batch_size=args.batch_size,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            drop_last=train_mode,
                            prefetch_factor=2,  # OOM error if this is too high
                            )
    
    return dataloader    
    
    
    
    




def build_model(args):
    '''Build ResidualLightAttention model'''

    # Build model with nn.Module
    args.model_params = {
        key: args.params[key] for key in 
        ['dim', 'dim', 'kernel_size', 'dropout', 'activation', 'res_blocks', 
         'random_seed']
        }
    args.model_params["out_dim"] = len(args.y_cols) 
    # once-hot encoding of the feature cols
    args.model_params["dim2"] = args.num_seq_features
    torch.manual_seed(args.model_params['random_seed'])  # Reproducibility
    model = torchmodels.ResidualLightAttention(**args.model_params)
    args.num_parameters = torchmodels.count_parameters(model)['FULL_MODEL']
    
    return model, args





def build_optimizer(model, args):
    '''Build optimizer for model'''
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.params['learning_rate'],
                           weight_decay=args.params['l2reg'])
    
    return optimizer
    

    



def save_model(model, optimizer, path, args):
    '''Save model and optimizer parameters to disk'''
    
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_params': args.model_params,
            'optimizer_state_dict': optimizer.state_dict(),
            },
        path)
    






def rename_state_dict(state_dict):
    '''Remove 'module' in keys of state_dict'''

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove 'module'
        new_state_dict[name] = v

    return new_state_dict






def load_model(model, path, args):
    '''Load model parameters from path'''
    
    checkpoint = torch.load(path, map_location='cpu')
    assert args.model_params == checkpoint['model_params'], \
        "Saved model params and current params are different"
    try:
        model_state_dict = checkpoint['model_state_dict']  
        model.load_state_dict(model_state_dict, strict=True)
    except:
        model_state_dict = rename_state_dict(model_state_dict)
        model.load_state_dict(model_state_dict, strict=True)            
        
    return model
    
    
    
    
    
    

def load_optimizer(optimizer, path, args):
    '''Load model parameters from path'''
    
    checkpoint = torch.load(path, map_location='cpu')
    assert args.model_params == checkpoint['model_params'], \
        "Saved model params and current params are different"
    try:
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dict)
    except:
        optimizer_state_dict = rename_state_dict(optimizer_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    
    
    return optimizer







def reduce_learning_rate(optimizer):
    '''Reduce learning rate by a factor of 0.5'''

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
    
    return optimizer





       
def root_mean_squared_error(ytrue, ypred, weight):
    '''Return the weighted mean squared error'''
    
    return torch.sqrt(torch.mean(((ytrue - ypred) ** 2) * weight))
    
    
    

       

def prepare_training(args):
    '''Prepare paths and objects for training'''
    
    
    # Paths/directories
    args.savepath = f'{args.savedir}/{args.model_name}'
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    args.recent_model = f'{args.savepath}/model_recent.pt' # Save model each epoch
    args.best_model = f'{args.savepath}/model_best.pt'     # Save model if error decreases
    args.log_file = f'{args.savepath}/log.csv'             # Write losses to file
    
    
    # Instantiate model 
    # Don't instantiate optimizer until model.to(device) is called
    model, args = build_model(args)
    
    
    # Model and training progress parameters
    if (not args.restart): # Train from scratch, don't restart
        
        # Write new log file
        with open(args.log_file, 'w') as logs:
            logs.write("epoch,train_loss,val_loss,best_val_loss,learning_rate,"\
                       "since_improved\n")
        
        # Progress parameters
        progress_params = {'epoch': 1, # Start from 1 to args.epoch + 1, not from 0
                           'best_val_loss': np.nan,
                           'since_improved': 0}
            
    else: # Restart training, don't train from scratch
        
        # Don't write new log files, assert that necessary files are in savepath
        error_msg = "Cannot restart training without "
        assert os.path.exists(args.log_file), error_msg + "progress file"
        assert os.path.exists(args.best_model), error_msg + "best_model"
        
        # Get progress parameters from log file
        logs = pd.read_csv(args.log_file, index_col=None, header=0)
        progress_params = {'epoch': logs['epoch'].values[-1],
                           'best_val_loss': logs['best_val_loss'].values[-1], 
                           'since_improved': logs['since_improved'].values[-1]}        

        # Load model from checkpoint
        model = load_model(model, args.best_model, args)
        
    return model, progress_params, args
     






def start_message(args):
    '''Description message at start of training'''
    
    if (args.verbose):
        print(args)
        print(f'INFO: MODEL_NAME = {args.model_name}')
        print(f'INFO: NUM_PARAMETERS = {args.num_parameters:.2e}')
        print(f'INFO: EPOCHS = {args.epochs}')
        print(f'INFO: NUM_NODES = {args.num_nodes}')
        print(f'INFO: GPUS_PER_NODE = {args.gpus_per_node}')
        print(f'INFO: NUM_GPUS = {args.world_size}')
        print(f'INFO: NUM_CPUS = {args.num_cpus}')
        print(f'INFO: NUM_WORKERS = {args.num_workers}')        
        print(f'INFO: SAVE_PATH = {args.savepath}')
        print()
        for key, value in args.params.items():
            print(f'PARAMS: {key}={value}')
        print()
        print('INFO: STARTING TRAINING')
        print()
    
        





def exit_message(epoch, args):
    '''Exit message at end of training'''
    
    if epoch >= args.epochs:
        print()
        print(f'INFO: Maximum training epochs ({args.epochs}) reached!')
        print('INFO: FINISHED TRAINING')
        
        return
    
    
    
    
    
    
def train_for_one_epoch(rank, trainloader, model, optimizer, epoch, args):
    '''Train model for one epoch'''
    
    _ = model.train()  # Set in training mode
    all_losses = []
    # temp fix
    device = "cuda:0"
    
    for i, data in enumerate(tqdm(trainloader)):

        if len(data) == 4:
            x, y, weight, mask = data
            x2 = None
        else:
            x, x2, y, weight, mask = data
            x2 = x2.to(device, dtype=torch.float32)
        x, y, weight, mask = x.to(device), y.to(device), weight.to(device), mask.to(device)

        optimizer.zero_grad()                          
        y_pred = model(x, x2=x2, mask=mask)[0]
        losses = []
        combined_loss = 0
        for j in range(args.model_params["out_dim"]):

            y_true = y[:, j]
            # only calculate the loss for non-nan rows
            loss = root_mean_squared_error(y_true[~torch.isnan(y_true)], 
                                           y_pred[~torch.isnan(y_true), j], 
                                           weight[~torch.isnan(y_true)]) 
            # give more weight to the ph_range weights
            if args.y_weights is not None:
                loss = torch.mul(loss, args.y_weights[j])
            if not torch.isnan(loss):
                combined_loss += loss 
            losses.append(loss) 

        _ = combined_loss.backward()
        _ = optimizer.step()
        all_losses.append(combined_loss.item())
    
        #if rank == 0:
        #    print(f'PROGRESS: step={i+1}, loss={combined_loss.item():.4f}, {losses = }', flush=True)

    return np.mean(all_losses)






def validate(rank, valloader, model, args):
    '''Validate model on validation set'''
    
    _ = model.eval()  # Set in validation mode
    all_losses = []
    # temp fix
    device = "cuda:0"

    with torch.no_grad():
        
        for i, data in enumerate(valloader):

            if len(data) == 4:
                x, y, weight, mask = data
                x2 = None
            else:
                x, x2, y, weight, mask = data
                x2 = x2.to(device, dtype=torch.float32)
            x, y, weight, mask = x.to(device), y.to(device), weight.to(device), mask.to(device)
            y_pred = model(x, x2=x2, mask=mask)[0]

            losses = []
            combined_loss = 0
            for j in range(args.model_params["out_dim"]):

                y_true = y[:, j]
                # only calculate the loss for non-nan rows
                loss = root_mean_squared_error(y_true[~torch.isnan(y_true)], 
                                               y_pred[~torch.isnan(y_true), j], 
                                               weight[~torch.isnan(y_true)]) 
                losses.append(loss) 
                if not torch.isnan(loss):
                    combined_loss += loss
        all_losses.append(combined_loss.item())
            
    return np.mean(all_losses)





def distributed_training(rank, world_size, args):
    '''Distributed training routine'''

    # Set up process groups for distributed training/inference
    #_ = configure_processes(rank, args)
    
    
    # Prepare dataloader
    traindata, args = get_dataset(args.trainseqs, 
                                  args.params['sample_method'], 
                                  args,
                                  **vars(args))
    trainloader = get_dataloader(rank, 
                                 world_size,
                                 traindata,
                                 True,
                                 args)
    valdata, args = get_dataset(args.valseqs, 
                                'bin_inverse', # Use bin_inv to reweight validataion data
                                args,
                                **vars(args))
    valloader = get_dataloader(rank,
                               world_size,
                               valdata, 
                               False,
                               args)
    
    # Prepare paths and model objects for training
    model, progress_params, args = prepare_training(args)
    model = model.to(rank)  # Move model to GPU device in DDP
    if rank == 0:
        start_message(args)


    
    # Build optimizer 
    optimizer = build_optimizer(model, args)
    if args.restart:
        optimizer = load_optimizer(optimizer, args.best_model, args)
    

    ## Wrap the model as a DistributedDataParallel object
    #model = DistributedDataParallel(model, 
    #                                device_ids=[rank],
    #                                output_device=rank,
    #                                find_unused_parameters=True)
    
    
    # Training epochs
    start_epoch = progress_params['epoch']  # start_epoch >1 if restarting
    stop_epoch = args.epochs + 1
    if rank == 0:
        _ = exit_message(start_epoch, args)
        if (args.restart):
            print()
            print(f"INFO: Restarting model for epoch {start_epoch + 1}")    
            print()
    
        
    
    model = model.to("cuda:0")
    print("\nmodel")
    print(model)
    # Training iteration
    for epoch in trange(start_epoch, stop_epoch):
        
        #_ = trainloader.sampler.set_epoch(epoch) # For distributed batches
        learning_rate = optimizer.param_groups[0]['lr']
        start_time = time.time()
        train_loss = train_for_one_epoch(rank, trainloader, model, optimizer, epoch, args)
        val_loss = validate(rank, valloader, model, args)
        end_time = time.time()
        total_time = end_time - start_time 

        
        # Writing and printing, only if gpu is master
        if rank == 0:
            
            # Save model and update progress parameters
            _ = save_model(model, optimizer, args.recent_model, args) # Save most recent model
            #epoch_model = args.recent_model.replace('_recent.pt', f'_epoch_{epoch}.pt')
            #_ = save_model(model, optimizer, epoch_model, args) # Save separate model after each epoch
            
            if not (val_loss >= progress_params['best_val_loss']):
                # Save model as best model, if validation loss has improved
                _ = save_model(model, optimizer, args.best_model, args)
                progress_params['best_val_loss'] = val_loss
                progress_params['since_improved'] = 0 # Reset
            else:
                progress_params['since_improved'] += 1 
                
            
            # Reduce learning rate if performance hasn't improved
            if (progress_params['since_improved']>= args.reduce_lr_patience) and \
                (progress_params['since_improved'] % args.reduce_lr_patience == 0):
                    optimizer = reduce_learning_rate(optimizer)
                
            # Log progress
            with open(args.log_file, 'a') as logs:
                logs.write(f"{epoch},{train_loss},{val_loss},"\
                           f"{progress_params['best_val_loss']},"\
                           f"{learning_rate},"\
                           f"{progress_params['since_improved']}\n")
                    
            if args.verbose == 1:
                print()
                print(f'PROGRESS: epoch={epoch}, '\
                      f'train_loss={train_loss:.4f}, '\
                      f'val_loss={val_loss:.4f}, '\
                      f"best_val_loss={progress_params['best_val_loss']:.4f}, "\
                      f"learning_rate={learning_rate:.2e}, "\
                      f"since_improved={progress_params['since_improved']}, "\
                      f"time={total_time:.1f}s")
            
            # Exit if validation hasn't improved 
            if progress_params['since_improved'] >= args.stop_patience:
                print(f"INFO: Exiting, as validation loss has not improved since "\
                      f"{progress_params['since_improved']} epochs")
                print("INFO: FINISHED TRAINING")
                return
                
    
    # Exit training
    if rank == 0:
        _ = exit_message(epoch, args)
    return

                
            
            



def main():
    '''Main routine'''
    
    # Parse command-line training arguments
    args = create_parser()                
    
    # Configure GPUs for processes
    world_size, args = configure_cuda(args)        
    
    # Distributed training raining routine
    #_ = mp.spawn(distributed_training,
    #             nprocs=args.world_size,
    #             args=(args.world_size, args)
    #             )
    # I think this means that this is the main process
    rank = 0
    distributed_training(rank, world_size, args)




if __name__=='__main__':

    _ = main()


