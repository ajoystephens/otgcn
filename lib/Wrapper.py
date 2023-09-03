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
from scipy.stats import linregress

from scipy.stats import hmean
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
 
import sys
import os
from lib.Models import *
from lib.Transformer import *
from lib.utils import *



class ModelWrapper():
    def __init__(self,options):
        self.options = options.copy()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to_validate = False

        self.priv_group = 1
        self.pos_label = 1

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_logging(self, log_filepath,log_filename):
        self.log = setupLogging('FGO',log_filepath,log_filepath + '/' + log_filename)
        self.results_path_root = log_filepath
        
        self.log.info(f'====={" SETUP LOGGING " :=<85}')

        self.log.info(f'THIS_FILE: {os.path.basename(__file__)}')
        
        for key in self.options:
            self.log.info(f'\t{key}: {self.options[key]}')

        self.log.info(f'====={" SETUP LOGGING COMPLETE " :=<85}')

    def load_and_split_source(self,source,fold_assignments, fold):
        self.source = source

        self.log.info(f'====={" Configuring Masks " :=<85}')
        fa = np.array(fold_assignments)

        train_mask=np.array(fa!=fold)
        val_mask=np.array(fa==fold)
        test_mask=val_mask
        source_mask = train_mask.copy()
        target_mask = test_mask.copy()

        N_s = source_mask.sum()
        N_t = target_mask.sum()
        N = N_s + N_t

        self.log.info(f'Source Total: {N_s}')
        self.log.info(f'Target Total: {N_t}')

        self.log.info(f'Train Total: {train_mask.sum()}')
        self.log.info(f'Val Total:   {val_mask.sum()}')
        self.log.info(f'Test Total:  {test_mask.sum()}')

        self.to_validate = True

        self.log.info(f'====={" Preparing to load data " :=<85}')
        self.source = source
        self.log.info(f'SOURCE: {self.source}')
        self.log.info(f'====={" Retrieving SOURCE Data " :=<85}')
        A, X, P, Y, has_label_mask_s = load_fair_graph_data('./data/'+str(self.source)+'.mat',self.log)
        
        self.log.info(f'====={" Preparing data " :=<85}')  


        P_s = P[source_mask]
        P_t = P[target_mask]
 
        self.options['P_s'] = torch.tensor(P_s,dtype=torch.float).to(self.device)
        self.options['P_t'] = torch.tensor(P_t,dtype=torch.float).to(self.device)

        # X=sp.vstack((X_s, X_t))
        X = sp.lil_matrix(X).toarray()
        # X = preprocessing.normalize(X, axis=0)

        Y_single = Y.argmax(axis=1)
        self.Y_np = Y

        self.log.info(f'====={" Disconnecting Source and Target " :=<85}') 
        A = A.tolil()
        i_source = np.arange(N)[source_mask]
        i_target = np.arange(N)[target_mask]
        for i in i_source:
            for j in i_target:
                A[i,j]=0.0
                A[j,i]=0.0


        self.log.info(f'====={" Finalize Data " :=<85}')


        self.log.info(f'A shape: {A.shape}')
        self.log.info(f'X shape: {X.shape}')
        self.log.info(f'Y shape: {Y.shape}')


        self.log.info(f'N: {N}')
        self.log.info(f'N_s: {N_s}')
        self.log.info(f'N_t: {N_t}')

        data_y = torch.tensor(Y_single,dtype=torch.long).to(self.device)
        data_x = torch.tensor(X,dtype=torch.float).to(self.device)
        e_ind, e_wei = from_scipy_sparse_matrix(A)
    
        self.data = Data(x=data_x,edge_index=e_ind,num_nodes=N,y=data_y)
        self.data.source_mask = torch.tensor(source_mask, dtype=torch.bool)
        self.data.target_mask = torch.tensor(target_mask, dtype=torch.bool)
        self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        self.data.validate(raise_on_error=True)
        if torch.cuda.is_available(): self.data.cuda(device=self.device)

        self.N_s = N_s
        self.N_t = N_t


    def load_data(self,source,target):
        self.log.info(f'====={" Preparing to load data " :=<85}')
        self.source = source
        self.target = target
        self.log.info(f'SOURCE: {self.source}')
        self.log.info(f'TARGET: {self.target}')
        self.log.info(f'====={" Retrieving SOURCE Data " :=<85}')
        A_s, X_s, P_s, Y_s, has_label_mask_s = load_fair_graph_data('./data/'+str(self.source)+'.mat',self.log)
        self.log.info(f'====={" Retrieving TARGET Data " :=<85}')
        A_t, X_t, P_t, Y_t, has_label_mask_t = load_fair_graph_data('./data/'+str(self.target)+'.mat',self.log)
        self.options['P_s'] = torch.tensor(P_s,dtype=torch.float).to(self.device)
        self.options['P_t'] = torch.tensor(P_t,dtype=torch.float).to(self.device)

        self.log.info(f'====={" Combine Data " :=<85}')
        N_s = A_s.shape[0]
        N_t = A_t.shape[0]
        N = N_s + N_t

        A=sp.lil_matrix((N_s+N_t,N_s+N_t),dtype=np.float32)
        A[0:N_s,0:N_s]=A_s
        A[-N_t:,-N_t:]=A_t
        # A = A.todense()

        X=sp.vstack((X_s, X_t))
        X = sp.lil_matrix(X).toarray()
        # X = preprocessing.normalize(X, axis=0)


        Y=np.concatenate((Y_s, Y_t),axis=0)
        Y_single = Y.argmax(axis=1)
        self.Y_np = Y



        self.log.info(f'A shape: {A.shape}')
        self.log.info(f'X shape: {X.shape}')
        self.log.info(f'Y shape: {Y.shape}')


        self.log.info(f'N: {N}')
        self.log.info(f'N_s: {N_s}')
        self.log.info(f'N_t: {N_t}')

        data_y = torch.tensor(Y_single,dtype=torch.long).to(self.device)
        data_x = torch.tensor(X,dtype=torch.float).to(self.device)
        # e_ind, e_wei = from_scipy_sparse_matrix(A*self.options['alpha'])
        e_ind, e_wei = from_scipy_sparse_matrix(A)

        source_mask = np.zeros(N,dtype=bool)
        source_mask[np.arange(N_s)] = True
        target_mask=np.concatenate((
            np.array(np.zeros(N_s), dtype=bool), 
            np.array(np.ones(N_t), dtype=bool)), axis=0)
    
        self.data = Data(x=data_x,edge_index=e_ind,num_nodes=N,y=data_y)
        self.data.source_mask = torch.tensor(source_mask, dtype=torch.bool)
        self.data.target_mask = torch.tensor(target_mask, dtype=torch.bool)

        self.N_s = N_s
        self.N_t = N_t


    def set_masks_by_fold(self, fold_assignments, fold):
        self.log.info(f'====={" Configuring Train/Val/Test Masks " :=<85}')
        fa = np.array(fold_assignments)

        train_mask=np.concatenate((fa!=fold, np.zeros(self.N_t)),axis=0)
        val_mask=np.concatenate((fa==fold, np.zeros(self.N_t)),axis=0)
        test_mask=np.concatenate((np.zeros(self.N_s), np.ones(self.N_t)),axis=0)

        self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        self.data.validate(raise_on_error=True)
        if torch.cuda.is_available(): self.data.cuda(device=self.device)

        self.log.info(f'Train Total: {train_mask.sum()}')
        self.log.info(f'Val Total:   {val_mask.sum()}')
        self.log.info(f'Test Total:  {test_mask.sum()}')

        self.to_validate = True
    
    def set_masks_full_source_target(self):
        self.log.info(f'====={" Configuring Train/Val/Test Masks " :=<85}')
        N = self.N_s+self.N_t
        source_mask = np.zeros(N,dtype=bool)
        source_mask[np.arange(self.N_s)] = True
        target_mask=np.concatenate((
            np.array(np.zeros(self.N_s), dtype=bool), 
            np.array(np.ones(self.N_t), dtype=bool)), axis=0)

        train_mask=source_mask
        val_mask=np.zeros(N)
        test_mask=target_mask

        self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        self.data.validate(raise_on_error=True)
        if torch.cuda.is_available(): self.data.cuda(device=self.device)

        self.log.info(f'Train Total: {train_mask.sum()}')
        self.log.info(f'Val Total:   {val_mask.sum()}')
        self.log.info(f'Test Total:  {test_mask.sum()}')

        self.to_validate = False

    def initialize_model(self):
        self.model = GCNOT(self.data.num_features,self.Y_np.shape[1],
                         self.options,self.log,self.device)
        self.model = self.model.to(self.device)

    def pretrain(self, to_save_loss=False):
        self.log.info(f'====={" Pretrain GCN Model " :=<85}')
        train_losses = []
        val_losses = []
        val_f1 = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.model.optimizer

        train_mask = self.data.train_mask.clone().detach().cpu().numpy()
        val_mask = self.data.val_mask.clone().detach().cpu().numpy()
        y = self.data.y.clone().detach().cpu().numpy()

        best = {'val_loss':1000.0,'hmean':0.0,'macro_f1':0.0,'sp':1.0,'epoch':self.options['max_pretrain_epochs']}

        # model.train()
        for epoch in range(self.options['max_pretrain_epochs']+1):
            # if((epoch % 10 == 0) and TO_SAVE_HIDDEN):
            self.model.train()

            optimizer.zero_grad()
            out, out_softmax = self.model(self.data.x, self.data.edge_index,to_transform=False)
            
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            out_softmax = out_softmax.clone().detach().cpu().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options['clip'])
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                if self.to_validate:
                    val_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
                    tmp_mask_set = 'val'
                else: 
                    tmp_mask_set = 'train'
                pred_np = out_softmax.argmax(axis=1)
                f1 = self.get_eval_metric(pred_np,y,mask_set=tmp_mask_set)

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                if self.to_validate:
                    self.log.info(f'Epoch {epoch:>3} '
                        f'| Train Loss: {loss:.3f} '
                        f'| Val Loss: {val_loss:.2f} | Val F1: {f1:.2f} |')
                else:
                    self.log.info(f'Epoch {epoch:>3} '
                        f'| Train Loss: {loss:.3f} '
                        f'| Train Macro F1: {f1:.2f} |')


            train_losses.append(loss.item())
            if self.to_validate:
                val_losses.append(val_loss.item())
                val_f1.append(f1)

            if self.to_validate and (f1>best['macro_f1']):
                best = {
                    'epoch': epoch,
                    'train_loss': loss,
                    'val_loss': val_loss,
                    'macro_f1': f1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
        
        if self.to_validate:
            self.log.info(f'Best Epoch: {best["epoch"]:>3} '
                    f'| Train Loss: {best["train_loss"]:.3f} '
                    f'| Val Loss: {best["val_loss"]:.2f} | Val F1: {best["macro_f1"]:.2f} ')
            self.log.info('Setting model to best state.')
            self.initialize_model()
            self.model.load_state_dict(best['model_state_dict'])

        # save losses 
        if to_save_loss:
            dirpath = self.results_path_root+'/loss/pretrain/'
            save_np(train_losses, dirpath,'train_losses.csv')
            save_np(val_losses, dirpath,'val_losses.csv')
            save_np(val_f1, dirpath,'val_f1.csv')

        return best["epoch"]

    def train_ot(self, to_save_loss=False, to_save_hidden=False):
        self.log.info(f'====={" Train Fair OT GCN Model " :=<85}')
        train_ce_losses = []
        train_ot_losses = []
        train_losses = []
        val_ce_losses = []
        val_ot_losses = []
        val_losses = []
        val_eo = []
        val_f1 = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.model.optimizer

        self.model.transform_source_mask = self.data.source_mask
        self.model.transform_target_mask = self.data.target_mask
        self.model.labels = self.data.y

        train_mask = self.data.train_mask.clone().detach().cpu().numpy()
        val_mask = self.data.val_mask.clone().detach().cpu().numpy()
        y = self.data.y.clone().detach().cpu().numpy()

        best = {'val_loss':1000.0,'hmean':0.0,'macro_f1':0.0,'sp':1.0,'epoch':self.options['max_pretrain_epochs']}

        # model.train()
        for epoch in range(self.options['max_ot_epochs']):
            # if((epoch % 10 == 0) and TO_SAVE_HIDDEN):
            self.model.train()
            if (to_save_hidden and (epoch==0)): # save first (also last, but later)
                self.model.to_save_hidden = to_save_hidden
                self.model.hidden_dir = self.results_path_root+'/hidden'


            # Training
            optimizer.zero_grad()
            out, out_softmax = self.model(self.data.x, self.data.edge_index,
                                     to_transform=True)

            ce_loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            ot_loss = self.model.get_transport_loss(train_mask
                                                    )
            loss = ce_loss + self.options['theta'] * ot_loss

            out_softmax = out_softmax.clone().detach().cpu().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options['clip'])
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                if self.to_validate:
                    val_ce_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
                    val_ot_loss = self.model.get_transport_loss(val_mask)
                    val_loss = val_ce_loss + self.options['theta'] * val_ot_loss
                    tmp_mask_set = 'val'
                else: 
                    tmp_mask_set = 'train'
                pred_np = out_softmax.argmax(axis=1)
                f1 = self.get_eval_metric(pred_np,y,mask_set=tmp_mask_set)
            # with torch.no_grad():
            #     val_ce_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            #     val_ot_loss = self.model.get_transport_loss(val_mask)
            #     val_loss = val_ce_loss + self.options['theta'] * val_ot_loss
            #     pred_np = out_softmax.argmax(axis=1)
            #     gm,eo,f1 = self.get_eval_metric(pred_np,y,mask_set='val')

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                if self.to_validate:
                    self.log.info(f'Epoch {epoch:>3} '
                        f'| Train Loss: {loss:.3f} '
                        f'| Val Loss: {val_loss:.2f} | Val F1: {f1:.2f} |')
                else:
                    self.log.info(f'Epoch {epoch:>3} '
                        f'| Train Loss: {loss:.3f} | Train CE Loss: {ce_loss:.3f} | Train OT Loss: {ot_loss:.3f}'
                        f'| Train Macro F1: {f1:.2f} |')
                
                # self.log.info(f'Epoch {epoch:>3} '
                #     f'| Train Loss: {loss:.3f} '
                #     f'| Val Loss: {val_loss:.2f} | Val F1: {f1:.2f} | Val EO: {eo:.2f} | Val hmean: {gm:.2f}')

            self.model.to_save_hidden = False


            train_ce_losses.append(ce_loss.item())
            train_ot_losses.append(ot_loss.item())
            train_losses.append(loss.item())
            if self.to_validate:
                val_ce_losses.append(val_ce_loss.item())
                val_ot_losses.append(val_ot_loss.item())
                val_losses.append(val_loss.item())
                val_f1.append(f1)

            if self.to_validate and (f1>best['macro_f1']) and epoch >5:
                best = {
                    'epoch': epoch,
                    'train_loss': loss,
                    'val_loss': val_loss,
                    'macro_f1': f1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
        
        if self.to_validate:
            self.log.info(f'Best Epoch: {best["epoch"]:>3} '
                    f'| Train Loss: {best["train_loss"]:.3f} '
                    f'| Val Loss: {best["val_loss"]:.2f} | Val F1: {best["macro_f1"]:.2f} |')
            self.log.info('Setting model to best state.')
            self.initialize_model()
            self.model.load_state_dict(best['model_state_dict'])


        # save losses 
        if to_save_loss:
            dirpath = self.results_path_root+'/loss/ot_train/'
            save_np(train_ce_losses, dirpath,'train_ce_losses.csv')
            save_np(train_ot_losses, dirpath,'train_ot_losses.csv')
            save_np(train_losses, dirpath,'train_losses.csv')
            save_np(val_ce_losses, dirpath,'val_ce_losses.csv')
            save_np(val_ot_losses, dirpath,'val_ot_losses.csv')
            save_np(val_losses, dirpath,'val_losses.csv')
            save_np(val_eo, dirpath,'val_sp.csv')
            save_np(val_f1, dirpath,'val_f1.csv')
        # save_np(target_losses, dirpath,'target_losses.csv')
        return best['epoch']
    
    def test(self,to_save_hidden=False):
        self.log.info(f'====={" Test Fair OT GCN Model " :=<85}')

        self.model.to_save_hidden = to_save_hidden
        self.model.hidden_dir = self.results_path_root+'/hidden/test'

        to_transform = True

        self.model.eval()

        self.model.transform_source_mask = self.data.source_mask
        self.model.transform_target_mask = self.data.target_mask
        self.model.labels = self.data.y

        out, out_softmax = self.model(self.data.x, self.data.edge_index,to_transform=to_transform)
        out_softmax = out_softmax.clone().detach().cpu().numpy()
        pred = out_softmax.argmax(axis=1)
        # pred_np = pred.clone().detach().cpu().numpy()
        pred_np = pred
        y = self.data.y.clone().detach().cpu().numpy()
        train_mask_np = self.data.train_mask.clone().detach().cpu().numpy()
        val_mask_np = self.data.val_mask.clone().detach().cpu().numpy()
        test_mask_np = self.data.test_mask.clone().detach().cpu().numpy()
        acc = accuracy(pred_np[test_mask_np], y[test_mask_np])
        self.model.to_save_hidden = False

        self.log.info(f'Train CM:\n{confusion_matrix(y[train_mask_np],pred[train_mask_np])}')
        self.log.info(f'Val CM:\n{confusion_matrix(y[val_mask_np],pred[val_mask_np])}')
        self.log.info(f'Test CM:\n{confusion_matrix(y[test_mask_np],pred[test_mask_np])}')

        dirpath = self.results_path_root+'/results/'
        save_np(y, dirpath,'y.csv')
        save_np(pred_np, dirpath,'pred.csv')
        save_np(out_softmax, dirpath,'pred_prob.csv')
        save_np(train_mask_np, dirpath,'train_mask.csv')
        save_np(val_mask_np, dirpath,'val_mask.csv')
        save_np(test_mask_np, dirpath,'test_mask.csv')

        return pred_np,y
    
    def get_f1(self,pred_np,y,mask_set='train'):
        if mask_set == 'test':
            mask = self.data.test_mask.clone().detach().cpu().numpy()
        elif mask_set == 'val':
            mask = self.data.val_mask.clone().detach().cpu().numpy()
        else: #default to training
            mask = self.data.train_mask.clone().detach().cpu().numpy()
        
        micro_f1 = f1_score(y[mask], pred_np[mask], average='micro')
        self.log.info(f'{mask_set} micro f1: {micro_f1}')
        macro_f1 = f1_score(y[mask], pred_np[mask], average='macro')
        self.log.info(f'{mask_set} macro f1: {macro_f1}')

        return macro_f1, micro_f1
    
    def get_fair(self,pred_np,y,mask_set='train',priv_group=1, pos_label=1):
        if mask_set == 'test':
            mask = self.data.test_mask.clone().detach().cpu().numpy()
        elif mask_set == 'val':
            mask = self.data.val_mask.clone().detach().cpu().numpy()
        else: #default to training
            mask = self.data.train_mask.clone().detach().cpu().numpy()
        
        P_s = self.options['P_s'].clone().detach().cpu().numpy()
        P_t = self.options['P_t'].clone().detach().cpu().numpy()
        P=np.concatenate((P_s, P_t),axis=0)
        
        sp = getStatParity(y[mask], pred_np[mask], P[mask],
                           priv_group=priv_group, pos_label=pos_label)
        self.log.info(f'{mask_set} statistical parity: {sp}')

        eo = getEqualOpportunity(y[mask], pred_np[mask], P[mask],
                           priv_group=priv_group, pos_label=pos_label)
        self.log.info(f'{mask_set} equal opportunity: {eo}')

        return sp,eo

    def get_eval_metric(self,pred_np,y,mask_set='train'):
        if mask_set == 'test':
            mask = self.data.test_mask.clone().detach().cpu().numpy()
        elif mask_set == 'val':
            mask = self.data.val_mask.clone().detach().cpu().numpy()
        else: #default to training
            mask = self.data.train_mask.clone().detach().cpu().numpy()
        
        macro_f1 = f1_score(y[mask], pred_np[mask], average='macro')

        return macro_f1

    def finish(self):
        # self.log.close()
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
