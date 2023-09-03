import scipy
import scipy.sparse as sp
import scipy.io as sio
import logging
import os
from scipy.sparse import csc_matrix

from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import equal_opportunity_difference

import numpy as np

def load_graph_data(filepath,log=None):
    """
    A_s, X_s, y_train_s, train_mask_s = load_data('./data/dblpv7.mat')
    """
    if log is not None: log.info('*************** In load_graph_data ***************')
    if log is not None: log.info(f'\tFILEPATH: {filepath}')
    net = sio.loadmat(filepath)
    X, A, Y = net['attrb'], net['network'], net['group']
    if log is not None: log.info(f'\tX shape: {X.shape}')
    if log is not None: log.info(f'\tA shape: {A.shape}')
    if log is not None: log.info(f'\tY shape: {Y.shape}')
    if log is not None: log.info(f'\tY sum: {Y.sum()} (number of postive labels)')

    if not isinstance(X, sp.lil_matrix):
        X = sp.lil_matrix(X)

    if 'has_label' in net.keys(): 
        # log.info('Dataset indicates that some datapoints do not have labels.')
        # y_mask = net['has_label'][0]
        has_label_mask = (net['has_label']==1)[0]
        y_label = Y[has_label_mask]
        tmp =net['has_label']
        if log is not None: log.info(f'\tnet[has_label] shape: {tmp.shape}')
        if log is not None: log.info(f'\tnet[has_label] sum: {tmp.sum}')
        if log is not None: log.info(f'\ty_label shape: {y_label.shape}')
        if log is not None: log.info(f'\ty_label sum: {y_label.sum()}')
    else: 
        y_label = Y
        has_label_mask = np.array(np.ones(Y.shape[0]), dtype=bool)

    return A, X, Y, has_label_mask

def load_fair_graph_data(filepath,log=None):
    """
    A_s, X_s, y_train_s, train_mask_s = load_data('./data/dblpv7.mat')
    """
    if log is not None: log.info('*************** In load_graph_data ***************')
    if log is not None: log.info(f'\tFILEPATH: {filepath}')
    net = sio.loadmat(filepath)
    X, A, Y, P = net['attrb'], net['network'], net['group'], net['prot'].flatten()
    if log is not None: log.info(f'\tX shape: {X.shape}')
    if log is not None: log.info(f'\tA shape: {A.shape}')
    if log is not None: log.info(f'\tP shape: {P.shape}')
    if log is not None: log.info(f'\tY shape: {Y.shape}')
    if log is not None: log.info(f'\tY sum: {Y.sum()} (number of postive labels)')

    if not isinstance(X, sp.lil_matrix):
        X = sp.lil_matrix(X)

    if 'has_label' in net.keys(): 
        # log.info('Dataset indicates that some datapoints do not have labels.')
        # y_mask = net['has_label'][0]
        has_label_mask = (net['has_label']==1)[0]
        y_label = Y[has_label_mask]
        tmp =net['has_label']
        if log is not None: log.info(f'\tnet[has_label] shape: {tmp.shape}')
        if log is not None: log.info(f'\tnet[has_label] sum: {tmp.sum}')
        if log is not None: log.info(f'\ty_label shape: {y_label.shape}')
        if log is not None: log.info(f'\ty_label sum: {y_label.sum()}')
    else: 
        y_label = Y
        has_label_mask = np.array(np.ones(Y.shape[0]), dtype=bool)

    return A, X, P, Y, has_label_mask

def agg_tran_prob_mat(g, step):
    """aggregated K-step transition probality"""
    g = my_scale_sim_mat(g)
    g = csc_matrix.toarray(g)
    a_k = g
    a = g
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, g)
        a = a+a_k/k
    return a
def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w

def get_ppmi(A,step):
    """compute PPMI, given aggregated K-step transition probality matrix as input"""
    # compute k-step transition probability matrix
    A = A.copy()
    A = my_scale_sim_mat(A)
    A = csc_matrix.toarray(A)
    a_k = A
    a = A
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, A)
        a = a+a_k/k

    # compute ppmi

    np.fill_diagonal(a, 0)
    a = my_scale_sim_mat(a)
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p)*a)/col[None, :])
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0
    return ppmi

def get_folds(dataset,n_folds):
    filepath = './data/'+str(dataset)+'.mat'
    net = sio.loadmat(filepath)
    Y = net['group']
    N = Y.shape[0]
    folds = np.random.randint(low=0,high=n_folds, size=N)
    return(folds)


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def load_prot(filepath,log=None):
    """
    A_s, X_s, y_train_s, train_mask_s = load_data('./data/dblpv7.mat')
    """
    if log is not None: log.info('*************** In load_prot ***************')
    if log is not None: log.info(f'\tFILEPATH: {filepath}')
    net = sio.loadmat(filepath)
    P = net['prot']
    return P

def get_time_string(t):
    seconds = t 
    minutes = seconds/60
    hours = minutes/60
    txt = f'{seconds} sec ({minutes} min) ({hours} hours)'
    return(txt)

def setupLogging(log_name, log_filepath, log_filename):
    # check if file exists
    if not os.path.exists(log_filepath): os.makedirs(log_filepath)

    LOG = logging.getLogger(log_name)
    LOG.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    LOG.addHandler(ch)
    LOG.addHandler(fh)
    return(LOG)

def save_tensor(h, dirpath,filename):
    np_h = h.clone().detach().cpu().numpy()
    save_np(np_h, dirpath,filename)

def save_np(data, dirpath,filename):
    if not os.path.exists(dirpath): os.makedirs(dirpath)
    filepath = dirpath+'/'+filename
    np.savetxt(filepath, data, delimiter=",")


def getEqualOpportunity(label, pred, prot, priv_group, pos_label):

    r = equal_opportunity_difference(label, pred, prot_attr=prot,
                                      priv_group=priv_group, pos_label=pos_label)
    r = abs(r)  
    return(r)

def getStatParity(label, pred, prot, priv_group, pos_label):

    r = statistical_parity_difference(label, pred, prot_attr=prot,
                                      priv_group=priv_group, pos_label=pos_label)
    r = abs(r)
    return(r)