# CHANGES TO THIS VERSION
# - add pre-train method
# - add alt loss

import time
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import scipy.sparse as sp
import pickle

from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
 

import sys
import os
import shutil
from lib.Models import *
from lib.Transformer import *
from lib.utils import *

from lib.Wrapper import * 

# https://towardsdatascience.com/graph-attention-networks-in-python-975736ac5c0c

###############################################################################
###############################################################################
# INITIALIZE THINGS
###############################################################################
###############################################################################
argParser = ArgumentParser()
argParser.add_argument("-s", "--source", default='pokec_trim_n_s__norm', help="source dataset")
# argParser.add_argument("-t", "--target", default='pokec_trim_z_s__norm', help="target dataset")
argParser.add_argument("-ot", "--optimal_transport", default=0, type=int, help="if should use optimal transport")
# argParser.add_argument("-fair", "--fair", default=1, type=float, help="if should do fair transform")
argParser.add_argument("-cont", "--continue_previous", default=0, type=float, help="if should continue from past run")
args = argParser.parse_args()

T_SCRIPT_BEGIN = time.time()
SOURCE = args.source
# TARGET = args.target

METHOD = 'GCN'

TRANSFORMATION = 'ot_pretrain' if (args.optimal_transport == 1) else 'None'

# FAIR = args.fair == 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 1

N_FOLDS = 10 # number of folds

EVAL_PARAMETERS = {
    'learning_rate': [0.001,0.005,0.01],
    'h1': [64], # dim of first hidden layer
    'h2': [32], # dim of second hidden layer
    'weight_decay': [5e-4],
    'dropout': [0.0], # check for overfitting first, if overfit then can increase
    'max_pretrain_epochs': [100], # max number of epochs to pretrain
    'max_ot_epochs': [50], # max number of epochs to perform OT
    'lambda': [0.01,0.03,0.05],
    'theta': [15,20,25],
    'clip': [5], # clip gradient
    'theta': [10,30,50]
}
PRIV_GROUP = 1 # female=0, male=1
POS_LABEL = 1 # is autism: 1, not autism: 0


NOW_STR = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")

###############################################################################
###############################################################################
# SETUP
###############################################################################
###############################################################################

# setup logging
LOGGER_NAME = 'CV'
LOG_FILEPATH = ('log/cv/'+SOURCE)
# if TRANSFORMATION != 'None': LOG_FILEPATH += '_'+TRANSFORMATION
if args.continue_previous == 0: 
    if os.path.exists(LOG_FILEPATH):
        shutil.rmtree(LOG_FILEPATH)
LOG_FILENAME = f'/main__{NOW_STR}.log'
LOG = setupLogging(LOGGER_NAME,LOG_FILEPATH,LOG_FILEPATH+LOG_FILENAME)

LOG.info(f'====={" FILE SETUP " :=<85}')
LOG.info(f'SOURCE: {SOURCE}')
# LOG.info(f'TARGET: {TARGET}')
LOG.info(f'N_FOLDS: {N_FOLDS}')
LOG.info(f'DEVICE: {DEVICE}')
LOG.info(f'continue_previous: {args.continue_previous}')
LOG.info(f'====={" DONE FILE SETUP " :=<85}')


# setting initial seed
LOG.info(f'====={" SETTING INITAL SEED " :=<85}')
LOG.info(f'SEED: {SEED}')
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
LOG.info(f'====={" DONE SETTING INITAL SEED " :=<85}')

if args.continue_previous == 1:
    LOG.info(f'====={" RETRIEVING OPTION SETS " :=<85}')
    # get option sets
    with open(LOG_FILEPATH+'/options.p', 'rb') as file:
        saved = pickle.load(file)
    OPTION_SETS=saved['OPTION_SETS']
    best_eval_stat=saved['best_eval_stat']
    best_set=saved['best_set']
    LOG.info(f'====={" DONE RETRIEVING OPTION SETS " :=<85}')

    LOG.info(f'====={" RETRIEVING FOLD ASSIGMENTS " :=<85}')
    # get fold_assigments
    with open(LOG_FILEPATH+'/folds.p', 'rb') as file:
        fold_assignments = pickle.load(file)
        fold_assignments = fold_assignments['fold_assignments']
    LOG.info(f'====={" DONE RETRIEVING FOLD ASSIGMENTS " :=<85}')
else: 
    # generate set of parameters
    LOG.info(f'====={" COMPILING OPTION SET " :=<85}')
    for key in EVAL_PARAMETERS:
        LOG.info(f'\t{key}: {EVAL_PARAMETERS[key]}')

    OPTION_SETS = [{'done':False}]
    for param in EVAL_PARAMETERS:
        new_set = []
        for opt in OPTION_SETS:
            for val in EVAL_PARAMETERS[param]:
                new_opt = opt.copy()
                new_opt[param] = val
                new_set.append(new_opt)
        OPTION_SETS = new_set.copy()
    LOG.info(f'Total Sets: {len(OPTION_SETS)}')
    LOG.info(f'====={" DONE COMPILING OPTION SET " :=<85}')

    # folds
    LOG.info(f'====={" ASSIGNING FOLDS " :=<85}')
    fold_assignments = get_folds(SOURCE,N_FOLDS)
    LOG.info(f'Fold Assignment Shape: {fold_assignments.shape}')
    unique, counts = np.unique(fold_assignments, return_counts=True)
    fold_counts = dict(zip(unique, counts))
    LOG.info(f'Fold Assignment Counts: \n{fold_counts}')
    # save off fold assignments 
    with open(LOG_FILEPATH+'/folds.p', 'wb') as file:
        pickle.dump({'fold_assignments':fold_assignments}, file)
    LOG.info(f'====={" DONE ASSIGNING FOLDS " :=<85}')

    best_eval_stat = 0
    best_set = {}

###############################################################################
###############################################################################
# CROSS FOLD VALIDATION
###############################################################################
###############################################################################

os_cnt = 1
os_cnt_str = f'{os_cnt:03}'

for option_set in OPTION_SETS:
    if option_set['done']: 
        os_cnt +=1
        os_cnt_str = f'{os_cnt:03}'
        continue # do not repeat already processed sets

    t_set_begin = time.time()
    LOG.info(f'====={" BEGINNING OPTION SET ("+os_cnt_str+")" :=<85}')
    for key in option_set:
        LOG.info(f'\t{key}: {option_set[key]}')
    
    total_pe = 0
    total_ot = 0 
    total_val_f1 = 0.0
    total_val_sp = 0.0
    total_eval_stat = 0.0
    for i_fold in range(N_FOLDS):
        t_fold_begin = time.time()
        LOG.info(f'-----{" BEGINNING FOLD SET ("+os_cnt_str+":"+str(i_fold)+")" :-<85}')
        model_wrapper = ModelWrapper(option_set)
        model_wrapper.priv_group = PRIV_GROUP
        model_wrapper.pos_label = POS_LABEL

        # setup sub logger
        run_path = f'{LOG_FILEPATH}/{os_cnt_str}/{i_fold}'
        LOG.info(f'Saving details to directory: \n{run_path}')
        model_wrapper.setup_logging(run_path,'this.log')

        # other setup
        model_wrapper.set_seed(SEED)

        # load data & setup fold
        model_wrapper.load_and_split_source(SOURCE,fold_assignments, i_fold)

        # initialize
        model_wrapper.initialize_model()

        try:
            # pretrain
            LOG.info(f'Begin Pretrain')
            pre_epochs = model_wrapper.pretrain(to_save_loss=True)
            total_pe += pre_epochs
            LOG.info(f'Pretrain Complete, {pre_epochs} epochs')

            # train OT
            LOG.info(f'Begin OT Train')
            ot_epochs = model_wrapper.train_ot(to_save_loss=True, to_save_hidden=False)
            total_ot += ot_epochs
            LOG.info(f'OT Train Complete, {ot_epochs} epochs')

            # get test results
            pred, label = model_wrapper.test(to_save_hidden=False)

            val_sp, val_eo = model_wrapper.get_fair(pred,label,mask_set='test',
                                                                priv_group=PRIV_GROUP, pos_label=POS_LABEL)
            f1 = model_wrapper.get_eval_metric(pred,label,mask_set='test')
            total_eval_stat += f1
            total_val_sp+=val_sp

            val_macro_f1, val_micro_f1 = model_wrapper.get_f1(pred,label,mask_set='test')
            total_val_f1+=val_macro_f1
            # wrap up
            model_wrapper.finish()
        except:
            LOG.error('CAUGHT ERROR')
            total_eval_stat += 0.0
            total_val_sp += 1.0
            total_val_f1+=0.0
            model_wrapper.finish()

        t = time.time()-t_fold_begin
        LOG.info(f'fold time: {t}s ({t/60}m)')


    # combine all results
    mean_pe = total_pe/N_FOLDS
    option_set['mean_pe'] = mean_pe
    LOG.info(f'Mean Pretrain Epochs {mean_pe}:')

    mean_ot = total_ot/N_FOLDS
    option_set['mean_ot'] = mean_ot
    LOG.info(f'Mean OT Train Epochs {mean_ot}:')

    mean_val_f1 = total_val_f1/N_FOLDS
    option_set['mean_val_f1'] = mean_val_f1
    LOG.info(f'Mean Val Macro F1 : {mean_val_f1}')

    mean_eval_stat = total_eval_stat/N_FOLDS
    option_set['mean_hmean'] = mean_eval_stat
    # LOG.info(f'Mean Eval Stat : {mean_eval_stat} (harmoic mean of macro f1 and (1-sp))')
    LOG.info(f'Mean Eval Stat : {mean_eval_stat} (harmoic mean of macro f1 and (1-eo))')


    # decide on best
    if mean_eval_stat > best_eval_stat:
        best_eval_stat=mean_eval_stat
        best_set = option_set.copy()
        LOG.info(f'NEW BEST SET! ({mean_eval_stat})')


    # save off current results
    option_set['done'] = True

    to_save ={}
    to_save['OPTION_SETS']=OPTION_SETS
    to_save['best_eval_stat']=best_eval_stat
    to_save['best_set']=best_set
    with open(LOG_FILEPATH+'/options.p', 'wb') as file:
        pickle.dump(to_save, file)

    os_cnt += 1
    os_cnt_str = f'{os_cnt:03}'

    t = time.time()-t_set_begin
    LOG.info(f'set time: {t}s ({t/60}m)')
    t = time.time()-T_SCRIPT_BEGIN
    LOG.info(f'script time so far: {t}s ({t/60}m) ({t/60/60}h)')

LOG.info(f'====={" CROSS FOLD COMPLETE ":=<85}')
LOG.info(f'Best Eval Stat: {best_eval_stat}')
LOG.info(f'Best Set: ')
for key in best_set:
    LOG.info(f'\t{key}: {best_set[key]}')

t = time.time()-T_SCRIPT_BEGIN
LOG.info(f'total script time: {t}s ({t/60}m) ({t/60/60}h)')
