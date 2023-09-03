import numpy as np
import torch 
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

import os

from lib.Transformer import OptimalTransport,printOT_Diff,printOT_2D

def save_tensor(h, dirpath,filename):
    if not os.path.exists(dirpath): os.makedirs(dirpath)
    filepath = dirpath+'/'+filename
    np_h = h.clone().detach().cpu().numpy()
    np.savetxt(filepath, np_h, delimiter=",")

 

class GCNOT(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_out,options,log,device):
        super().__init__()
        self.options = options
        self.gcn1 = GCNConv(dim_in, self.options['h1'])
        self.gcn2 = GCNConv(self.options['h1'], self.options['h2'])
        self.lin1 = Linear(dim_in, self.options['h1'])
        self.lin2 = Linear(self.options['h1'], self.options['h2'])
        self.lin = Linear(self.options['h2']*2, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = log, device=device)
        self.transform_source_mask = torch.zeros(10).to(device)
        self.transform_target_mask = torch.zeros(10).to(device)
        self.labels = torch.zeros(10).to(device) # used only in OT to print 
        self.log = log
        self.to_print_transform = False
        self.print_path = 'test.png'
        self.device = device

        self.to_save_hidden = False
        self.hidden_dir = ''
        # self.loss_add = 0


    def forward(self, x, edge_index,to_transform=False,to_save_transform=False,transform_path='/'):
        graph_h = self.gcn1(x, edge_index)
        graph_h = torch.relu(graph_h)
        if self.options['dropout'] >0:
            graph_h = F.dropout(graph_h, p=self.options['dropout'], training=self.training)
        graph_h = self.gcn2(graph_h, edge_index)
        graph_h = torch.relu(graph_h)
        if self.to_save_hidden: save_tensor(graph_h, self.hidden_dir,'pre_graph.csv')

        lin_h = self.lin1(x)
        lin_h = torch.relu(lin_h)
        lin_h = self.lin2(lin_h)
        lin_h = torch.relu(lin_h)
        if self.to_save_hidden: save_tensor(lin_h, self.hidden_dir,'pre_linear.csv')

        # print(f'lin_h: {lin_h.shape}')
        # print(f'graph_h: {graph_h.shape}')
        h = torch.cat((lin_h,graph_h),1)
        # print(f'concat: {h.shape}')

        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h,to_save_transform,transform_path)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.lin(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x,to_save_transform,transform_path):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        # self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')

        S_trans = self.transformer.transport(S,T,
            toComputeCost=True,costType='l2',toSaveTransport=to_save_transform,transportPath=transform_path)
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans

        return(x_trans)
    
    def get_transport_loss(self, mask):
        # loss = self.transformer.compute_wass_loss(mask)
        loss = self.transformer.compute_cuturi_sinkhorn_loss(mask)
        return(loss)

