import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.ticker import NullFormatter
from sklearn import manifold
import os

def save_tensor(h, dirpath,filename):
    if not os.path.exists(dirpath): os.makedirs(dirpath)
    filepath = dirpath+'/'+filename
    np_h = h.clone().detach().cpu().numpy()
    np.savetxt(filepath, np_h, delimiter=",")

class OptimalTransport():
    def __init__(self, ot_lambda=10, maxIter=10000, stopThr=1e-07, logger=None, nanHunt=True,device='cpu'):
        self.nanHunt = nanHunt
        self.log = logger
        self.l = ot_lambda
        self.maxIter=maxIter
        self.stopThr=stopThr
        # self.maxIter=1000
        # self.stopThr=0.2
        self.C = None
        self.device=device

    # ----------------------------------------------------------------------------------------
    # ----- PUBLIC
    # ----------------------------------------------------------------------------------------
    def transform(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=False,costScale=50,transformType='basic', 
        fair=False,alpha_s=1,is_prot_s=np.ones(5),alpha_t=1,is_prot_t=np.ones(5)):
    # def transform(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=False,transformType='basic', fair=False,target_marginal=np.ones(5)):
        error = 0
        error, S = self._cleanInput(S)
        error, T = self._cleanInput(T)

        if toComputeCost | (self.C is None):
            self._computeCostMatrix(S,T,costType,normalizeCost,costScale)

            if torch.isnan(self.C).any() or torch.isinf(self.C).any():
                msg = 'Cost matrix contains numerical errors'
                if self.log is None: print('WARNING: '+msg)
                else: self.log.error(msg)

        s = torch.ones(S.shape[0]).to(self.device)/S.shape[0] # use uniform distribution
        t = torch.ones(T.shape[0]).to(self.device)/T.shape[0] # use uniform distribution
        
        if fair:
            # error, t = self._cleanInput(target_marginal)
            s = self._compute_marginal(is_prot_s,alpha_s)
            t = self._compute_marginal(is_prot_t,alpha_t)
    

        K=torch.exp(self.C/(-self.l))
        if torch.isnan(K).any() or torch.isinf(K).any():
            msg = 'K matrix contains numerical errors, not performing transform'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
            return(S)

        error,u,v,d,P = self._sinkhorn(s,t)
        if error == 1:
            msg = 'error in sinkhorn, returning without transform'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
            return(S)
    
        # transformed = S.shape[0]*(P @ T)
        # print(P)
        error,transformed = self._transformSource(P,T,S,s,transformType=transformType)
        # print(transformed)
        self.P = P

        # loss = torch.sum(torch.mul(P,self.C))
        return(transformed)

    # like above but with loss
    def transport(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=True,costScale=100,transformType='basic', 
        alpha_s=1,is_prot_s=np.ones(5),alpha_t=1,is_prot_t=np.ones(5),toSaveTransport=False,transportPath='/'):
    # def transform(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=False,transformType='basic', fair=False,target_marginal=np.ones(5)):
        error = 0
        error, S = self._cleanInput(S)
        error, T = self._cleanInput(T)

        if toComputeCost | (self.C is None):
            self._computeCostMatrix(S,T,costType,normalizeCost,costScale)
            # if torch.isnan(self.C).any() or torch.isinf(self.C).any():
            #     msg = 'Cost matrix contains numerical errors'
            #     if self.log is None: print('WARNING: '+msg)
            #     else: self.log.error(msg)
            # self.log.error(f'cost cnt inf: {torch.sum(torch.isinf(self.C))} (after initial comp)')

        s = torch.ones(S.shape[0]).to(self.device)/S.shape[0] # use uniform distribution
        t = torch.ones(T.shape[0]).to(self.device)/T.shape[0] # use uniform distribution
    

        self.K=torch.exp(self.C/(-self.l))
        if torch.isnan(self.K).any() or torch.isinf(self.K).any():
            max_cost = torch.max(self.C)
            msg = 'K matrix contains numerical errors, not performing transform'
            msg += f'\n\t max cost: {max_cost}'
            msg += f'\n\t e^(max_cost/-l) cost: {torch.exp(max_cost/(-self.l))}'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
            return(S,0)

        error,u,v,d,P = self._sinkhorn(s,t)
        if error == 1:
            msg = 'error in sinkhorn, returning without transform'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
            return(S,0)
    
        # transformed = S.shape[0]*(P @ T)
        # print(P)
        error,transformed = self._transformSource(P,T,S,s,transformType=transformType)
        # print(transformed)

        if toSaveTransport: save_tensor(P, transportPath,'transport.csv')

        # loss = torch.sum(torch.mul(P,self.C))
        self.P = P
        return(transformed)
    
    # def fair_transport(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=True,costScale=100,transformType='basic', 
    #     fair=False,gamma=1,P_s=np.ones(5),P_t=np.ones(5),toSaveTransport=False,transportPath='/'):
    # # def transform(self,S,T,toComputeCost=True,costType='sq_l2',normalizeCost=False,transformType='basic', fair=False,target_marginal=np.ones(5)):
    #     error = 0
    #     error, S = self._cleanInput(S)
    #     error, T = self._cleanInput(T)

    #     self.g = gamma

    #     if toComputeCost | (self.C is None):
    #         self._computeCostMatrix(S,T,costType,normalizeCost,costScale)
    #         # if torch.isnan(self.C).any() or torch.isinf(self.C).any():
    #         #     msg = 'Cost matrix contains numerical errors'
    #         #     if self.log is None: print('WARNING: '+msg)
    #         #     else: self.log.error(msg)
    #         # self.log.error(f'cost cnt inf: {torch.sum(torch.isinf(self.C))} (after initial comp)')

    #     s = torch.ones(S.shape[0]).to(self.device)/S.shape[0] # use uniform distribution
    #     t = torch.ones(T.shape[0]).to(self.device)/T.shape[0] # use uniform distribution
    
    #     if fair: self.K = self._compute_fair_K(self.C,self.l,gamma,P_s,P_t)
    #     else: self.K=torch.exp(self.C/(-self.l))

    #     if torch.isnan(self.K).any() or torch.isinf(self.K).any():
    #         max_cost = torch.max(self.C)
    #         msg = 'K matrix contains numerical errors, not performing transform'
    #         msg += f'\n\t max cost: {max_cost}'
    #         msg += f'\n\t e^(max_cost/-l) cost: {torch.exp(max_cost/(-self.l))}'
    #         if self.log is None: print('WARNING: '+msg)
    #         else: self.log.error(msg)

    #         return(S,0)
    #     error,u,v,d,P = self._sinkhorn(s,t)
    #     if error == 1:
    #         msg = 'error in sinkhorn, returning without transform'
    #         if self.log is None: print('WARNING: '+msg)
    #         else: self.log.error(msg)
    #         return(S,0)
    
    #     # transformed = S.shape[0]*(P @ T)
    #     # print(P)
    #     error,transformed = self._transformSource(P,T,S,s,transformType=transformType)
    #     # print(transformed)

    #     if toSaveTransport: save_tensor(P, transportPath,'transport.csv')

    #     self.P = P
    #     # loss = torch.sum(torch.mul(P,self.C))
    #     return(transformed)
    

    # def compute_wass_loss(self,mask):
    #     n_s = self.P.shape[0]
    #     mask = mask[:n_s]
    #     # print(f'mask shape: {mask.shape}')
    #     m = torch.mul(self.P,self.C)
    #     # print(f'm shape: {m.shape}')
    #     m = m[mask,:]
    #     # print(f'm shape: {m.shape}')
    #     loss = torch.sum(m)
    #     # print(f'loss shape: {loss.shape}')
    #     return(loss)
    
    def compute_cuturi_sinkhorn_loss(self,mask):
        n_s = self.P.shape[0]
        mask = mask[:n_s]
        masked_P = self.P[mask,:]
        masked_C = self.C[mask,:]

        # print(f'mask shape: {mask.shape}')
        m = torch.mul(masked_P,masked_C)
        entropy = self.l*torch.mul(masked_P,torch.log(masked_P))# -1*lambda*entropy (since entropy negative we ignore first -1)

        loss = torch.sum(m+entropy)
        # print(f'loss shape: {loss.shape}')
        return(loss)
    
    # def compute_fair_sinkhorn_loss(self,mask):
    #     n_s = self.P.shape[0]
    #     mask = mask[:n_s]
    #     masked_P = self.P[mask,:]
    #     masked_C = self.C[mask,:]
    #     masked_R = self.R_prot[mask,:]
    #     masked_S = self.S_prot[mask,:]
    #     n_r = torch.sum(torch.sum(masked_R))
    #     n_s = torch.sum(torch.sum(masked_S))
    #     # print(f'mask shape: {mask.shape}')
    #     m = torch.mul(masked_P,masked_C)
    #     entropy = self.l*torch.mul(masked_P,torch.log(masked_P))# -1*lambda*entropy (since entropy negative we ignore first -1)

    #     fair = (masked_R/n_r) - (masked_S/n_s)
    #     fair = self.g*torch.mul(masked_P,fair)

    #     loss = torch.sum(m+entropy+fair)
    #     # print(f'loss shape: {loss.shape}')
    #     return(loss)

    
    # def _compute_fair_K(self,cost,lam,gamma,P_s,P_t):
    #     R_1 = torch.outer(P_s,P_t)
    #     R_0 = torch.outer(torch.abs(P_s-1),torch.abs(P_t-1))
    #     R = R_1+R_0
    #     S = torch.abs(R-1)
    #     n_r = torch.sum(torch.sum(R))
    #     n_s = torch.sum(torch.sum(S))

    #     self.R_prot = R
    #     self.S_prot = S

    #     # tmp = (R/n_r)-(S/n_s)
    #     # tmp = gamma * tmp
    #     # tmp = cost + tmp
    #     tmp = cost+(gamma*((R/n_r)-(S/n_s)))
    #     K=torch.exp(tmp/(-lam))
    #     return(K)


    # def _compute_marginal(self,is_prot,alpha=1):
    #     n = is_prot.shape[0]
    #     n_p = is_prot.sum()
    #     n_u = n-n_p
        
    #     # mu = np.multiply(alpha*is_prot,np.ones(n)) # weigh protected marginals by alpha
    #     mu = (alpha-1)*is_prot+torch.ones(n).to(self.device)
    #     mu = mu/((alpha*n_p)+n_u) # normalize to sum to one
        
    #     # print(n)
    #     # print(mu.shape)
    #     # print(mu)
        
    #     # mu = torch.from_numpy(mu).to(self.device).float() # convert to torch
    #     mu = mu.to(self.device).float() # convert to torch
        
    #     return(mu)

    # def simple_transform(self,S,T,toComputeCost=True,costType='simple'):
    #     error = 0

    #     s_bins = self._getBins(S)
    #     s_probDens = np.histogram(S,bins=s_bins,density=True)[0]
    #     s_bin_cnt = len(s_bins)-1

    #     t_bins = self._getBins(T)
    #     t_probDens = np.histogram(T,bins=t_bins,density=True)[0]
    #     t_bin_cnt = len(t_bins)-1

    #     if toComputeCost | (self.C is None):
    #         self._computeCostMatrix(s_bins,t_bins,costType)
    #     # self.log.error(f'cost cnt inf: {torch.sum(torch.isinf(self.C))} (after compute)')

    #     s = s_probDens # use uniform distribution
    #     t = t_probDens # use uniform distribution
    
    #     error, s = self._cleanInput(s)
    #     error, t = self._cleanInput(t)
    #     error, self.C = self._cleanInput(self.C)

    #     # self.log.error(f'cost cnt inf: {torch.sum(torch.isinf(self.C))} (after clean)')


    #     K=torch.exp(self.C/(-self.l))
    #     if torch.isnan(K).any() or torch.isinf(K).any():
    #         msg = 'K matrix contains numerical errors, not performing transform'
    #         if self.log is None: print('WARNING: '+msg)
    #         else: self.log.error(msg)
    #         return(S)

    #     error,u,v,d,P = self._sinkhorn(s,t)
    #     if error == 1:
    #         msg = 'error in sinkhorn, returning without transform'
    #         if self.log is None: print('WARNING: '+msg)
    #         else: self.log.error(msg)
    #         return(S)
    
    #     # a_trans = (T.T @ np.ones(T.shape[0]))*10000
    #     P = P.clone().detach().cpu().numpy()
    #     transformed = (P.T @ np.ones(P.shape[0]))*S.shape[0]
    #     # transformed = S.shape[0]*(P @ T)
    #     return(transformed)

    # ----------------------------------------------------------------------------------------
    # ----- SINKHORN
    # ----------------------------------------------------------------------------------------
    def _sinkhorn(self,s,t):
        # print(s.shape)
        error = 0
        i=torch.where(s>0)[0]
        s=s[i]
        C=self.C[i,:]
        eps = torch.tensor(1e-11)
        # print(s.shape)

        # K=torch.exp(self.C/(-self.l))
        K=self.K.to(torch.float)


        u=torch.ones(len(s)).to(self.device)/len(s)
        # v=torch.ones(len(t))/len(t)
        T=torch.ones(C.shape).to(self.device)
        tmp = torch.diag(1/s)
        tmp = tmp.to(torch.float)
        K_tilde = torch.matmul(tmp,K)
        
        u_diff=1
        i=0
        cnt=1

        while (u_diff>self.stopThr and cnt<self.maxIter):
            # v_new=t/(K.T @ u)
            # u_new= 1 / (K_tilde @ v_new)
            # self.log.error(f't sum: {torch.sum(t)}')
            # self.log.error(f'K_tilde sum: {torch.sum(K_tilde)}')
            # self.log.error(f'C min: {torch.min(self.C)}')
            # self.log.error(f'C max: {torch.max(self.C)}')
            # self.log.error(f'C/l min: {torch.min(self.C/(-self.l))}')
            # self.log.error(f'C/l max: {torch.max(self.C/(-self.l))}')
            # self.log.error(f'-C/l cnt inf: {torch.sum(torch.isinf(self.C/(-self.l)))}')
            # temp = (K.T @ u)
            # self.log.error(f'cost cnt inf: {torch.sum(torch.isinf(self.C))}')
            # self.log.error(f'cost cnt equal zero: {torch.sum(self.C==0)}')
            # self.log.error(f'(K.T) cnt equal zero: {torch.sum(K.T==0)}')
            # self.log.error(f'(u) cnt equal zero: {torch.sum(u==0)}')
            # self.log.error(f'(K.T @ u) cnt equal zero: {torch.sum(temp==0)}')
            # self.log.error(f'(K.T @ u) sum: {torch.sum(K.T @ u)}')
            # self.log.error(f'K.T shape: {K.T.shape}')
            # self.log.error(f'u shape: {u.shape}')
            # self.log.error(f'(K.T @ u) shape: {temp.shape}')
            # temp = (t/(K.T @ u))
            # self.log.error(f'(t/(K.T @ u)) sum: {torch.sum((t/(K.T @ u)))}')
            # self.log.error(f'(t/(K.T @ u)) sum inf: {torch.sum(torch.isinf(temp))}')
            # self.log.error(f'(K_tilde @ (t/(K.T @ u))) sum: {torch.sum((K_tilde @ (t/(K.T @ u))))}')
            u_new= 1 / (K_tilde @ (t/((K.T @ u)+eps)) + eps)
            
            if torch.isnan(u_new).any() or torch.isinf(u_new).any():
                msg = f'numerical errors, returning last stable solution (cnt: {cnt})'
                if self.log is None: print('WARNING: '+msg)
                else: self.log.error(msg)
                break
                
            u_diff = torch.norm(u-u_new).item()
            # v=v_new
            u=u_new
            cnt+=1
            
            if cnt%1000==0:
                # print('WARNING: long running sinkhorn ('+str(cnt)+'), u_diff: '+str(u_diff))
                msg = 'long running sinkhorn ('+str(cnt)+'), u_diff: '+str(u_diff)
                if self.log is None: print('WARNING: '+msg)
                else: self.log.warning(msg)

        v=t/(K.T @ u)
        d = torch.sum(u * ((K*C) @ v))
        T = self.getTransportMatrix_torch(u,K,v)
        # if torch.isnan(T).any() or torch.isinf(T).any(): print('WARNING: T has numerical errors')
        if cnt>= 1000: 
            msg = 'large sinkhorn itter ('+str(cnt)+')'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.warning(msg)

        return(error,u,v,d,T)

    def getTransportMatrix_torch(self,u,K,v):
        return(torch.diag(u) @ K @ torch.diag(v))
    # ----------------------------------------------------------------------------------------
    # ----- TRANSFORMATION MANAGEMENT
    # ----------------------------------------------------------------------------------------
    def _transformSource(self,P,X_t,X_s,mu_s,transformType='marginal'):
        error = 0
        X_transformed = torch.ones(X_s.size()).to(self.device)
        if (transformType == 'basic'):
            X_transformed = self._transform_basic(P,X_t,X_s)
        elif (transformType == 'barycentric'):
            X_transformed = self._transform_barycentric(P,X_t,X_s)
        elif (transformType == 'marginal'):
            X_transformed = self._transform_marginal(P,X_t,mu_s)
        else:
            error=1
            self.log.error('Unexpected type of data transform: '+str(transformType))
        
        return(error,X_transformed)
    def _transform_basic(self,P,X_t,X_s):
        # transformed = (P.T @ np.ones(P.shape[0]))*S.shape[0]
        transformed = X_s.shape[0]*(P @ X_t)
        return(transformed)
    def _transform_barycentric(self,P,X_t,X_s):
        transformed = P / torch.sum(P,axis=1)[:,None]
        transformed[~ torch.isfinite(transformed)]=0
        transformed = transformed @ X_t
        return(transformed)
    def _transform_marginal(self,P,X_t,mu_s):
        A = torch.diag(1.0/mu_s)
        transformed = A @ P @ X_t
        return(transformed)

    # ----------------------------------------------------------------------------------------
    # ----- COST MANAGEMENT
    # ----------------------------------------------------------------------------------------

    def _computeCostMatrix(self,S,T,costType='sq_l2',normalize=False,costScale=100):
        error = 0
        if (costType == 'l2'):
            self.C = self._costType_l2(S,T)
        elif (costType == 'sq_l2'):
            self.C = self._costType_square_l2(S,T)
        elif (costType == 'simple'):
            self.C = self._costType_simple(S,T)
        else:
            error=1
            self.log.error('Unexpected type of cost calculation: '+str(costType))
        
        if normalize:
            self._normalizeCost(costScale)
        return(error)

    def _costType_square_l2(self,S,T):
        # m=S.shape[0]
        # n=T.shape[0]
        # C = torch.ones((m,n)).to(self.device)
        
        # for i in range(m):
        #     D = torch.outer(torch.ones(n).to(self.device),S[i,:])
        #     D = D - T
        #     C[i,:] = torch.linalg.norm(D, axis=1)
        
        # C = torch.square(C)
        C = torch.square(torch.cdist(S,T))
        return(C)

    def _costType_l2(self,S,T):
        m=S.shape[0]
        n=T.shape[0]
        C = torch.ones((m,n)).to(self.device)
        
        for i in range(m):
            D = torch.outer(torch.ones(n).to(self.device),S[i,:])
            D = D - T
            C[i,:] = torch.linalg.norm(D, axis=1)
        return(C)
    
    def _normalizeCost(self,scale):
        # msg = f'Scaling Cost: {scale}, max: {torch.max(self.C)} '
        # if self.log is None: print('INFO: '+msg)
        # else: self.log.info(msg)
        self.C /= scale

    # def getCostMatrix(S,T):
    #     m=S.shape[0]
    #     n=T.shape[0]
    #     C = np.empty((m,n))
        
    #     for i in range(m):
    #         D = np.outer(np.ones(n),S[i,:])
    #         D = D - T
    #         C[i,:] = np.linalg.norm(D, axis=1)
    #     return(C)

    # get cost matrix
    # - using the difference between the values
    def _costType_simple(self,sourceBins, targetBins):
        A=np.outer(np.ones(len(sourceBins[:-1])),targetBins[:-1])
        B=np.outer(sourceBins[:-1],np.ones(len(targetBins[:-1])))

        C = np.abs(A-B)
        return(C)

    # ----------------------------------------------------------------------------------------
    # ----- UTILITY
    # ----------------------------------------------------------------------------------------

    def _cleanInput(self,input_array):
        error = 0 # error flag to indicate if data is in good condition to proceed
        if input_array is None:
            msg = 'Input is None'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
        elif type(input_array) is np.ndarray:
            msg = 'Input is Numpy, converting to tensors....'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.debug(msg)
            input_array = torch.from_numpy(input_array).to('cpu').float()
        
        # should also check for nans here
        if self.nanHunt & torch.isnan(input_array).any():
            msg = 'Nan in Input'
            if self.log is None: print('WARNING: '+msg)
            else: self.log.error(msg)
            error = 1

        return(error, input_array)

    def _getBins(self,v):
        v_min=int(np.floor(v.min()))
        v_max=int(np.ceil(v.max()))

        bins = [*range(v_min,v_max+1)]
        return(bins)

def printOT(source,target,new_source,log,is_top_target=True,save_fig=True,title=None,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    
    error = 0
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))

    n_components=2
    perplexity=5

    (fig, subplots) = plt.subplots(1, 2, figsize=(15, 5))


    source = group == 1
    target = group == 2
    transform = group == 3


    # red = y == 2
    # green = y == 1

    # t0 = time()
    tsne = manifold.TSNE(
        n_components=n_components,
        init="random",
        random_state=10,
        perplexity=100,
        learning_rate="auto",
        n_iter=2000,
    )
    X_new = tsne.fit_transform(X)
    # t1 = time()

    if title is not None:
        st = fig.suptitle(title, fontsize="x-large")

    ax = subplots[0]
    # print("perplexities=%d" % (perplexity))
    ax.set_title('source (blue), target (red)')
    if(is_top_target):
        ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
        ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
    else:
        ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
        ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[green, 0], X_new[green, 1], c="g")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    ax = subplots[1]
    ax.set_title('transform (green), target (red)')
    # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    if(is_top_target):
        ax.scatter(X_new[transform, 0], X_new[transform, 1], c="b")
        ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
    else:
        ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
        ax.scatter(X_new[transform, 0], X_new[transform, 1], c="g")
    # ax.scatter(X_new[transform, 0], X_new[transform, 1], c="g")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()

def printOT_Fair(source,target,new_source,
    prot_source,prot_target,
    label_source,label_target,
    log,title=None,is_top_target=True,save_fig=True,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    
    error = 0
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))

    label_source = label_source.clone().detach().cpu().numpy()
    label_source = np.array([label_source != 0]).flatten()
    label_target = label_target.clone().detach().cpu().numpy()
    label_target = np.array([label_target != 0]).flatten()
    nlabel_source = np.logical_not(label_source)
    nlabel_target = np.logical_not(label_target)

    prot_source = prot_source.clone().detach().cpu().numpy()
    # prot_source = np.reshape(prot_source.shape[1], 6)
    prot_source = np.array([prot_source != 0]).flatten()
    prot_target = prot_target.clone().detach().cpu().numpy()
    prot_target = np.array([prot_target != 0]).flatten()
    unprot_source = np.logical_not(prot_source)
    unprot_target = np.logical_not(prot_target)

    n_components=2
    perplexity=5

    (fig, subplots) = plt.subplots(1, 2, figsize=(20, 8))

    if title is not None:
        st = fig.suptitle(title, fontsize="x-large")


    source = group == 1
    target = group == 2
    transform = group == 3


    # red = y == 2
    # green = y == 1

    # t0 = time()
    log.info('Begin fit TSNE...')
    tsne = manifold.TSNE(
        n_components=n_components,
        init="random",
        random_state=10,
        perplexity=100,
        learning_rate="auto",
        n_iter=2000,
    )
    X_new = tsne.fit_transform(X)
    log.info('End fit TSNE...')
    # log.info('X_new shape:  '+str(X_new.shape))
    # log.info('X_new source shape:  '+str(X_new[source, 0].shape))
    # log.info('prot source mask shape:  '+str(prot_source.shape))
    # log.info('unprot source mask shape:  '+str(unprot_source.shape))
    # log.info('X_new unprot source mask shape:  '+str(X_new[source, 0][unprot_source].shape))
    # t1 = time()

    ax = subplots[0]
    # print("perplexities=%d" % (perplexity))
    ax.set_title('source (blue), target (red)')
    if is_top_target:
        ax.scatter(X_new[source, 0][unprot_source], X_new[source, 1][unprot_source & label_source], c="lightblue", marker='X')
        ax.scatter(X_new[source, 0][prot_source], X_new[source, 1][prot_source], c="blue", marker='x')
        ax.scatter(X_new[target, 0][unprot_target], X_new[target, 1][unprot_target], c="lightcoral",marker='X')
        ax.scatter(X_new[target, 0][prot_target], X_new[target, 1][prot_target], c="red",marker='x')

        ax.scatter(X_new[source, 0][unprot_source], X_new[source, 1][unprot_source & nlabel_source], c="lightblue", marker='P')
        ax.legend(['source, unprot, true', 'source, prot','target, unprot','target, prot','source, unprot, false'])
    else:
        legend =[]
        ax.scatter(X_new[target, 0][unprot_target & label_target], X_new[target, 1][unprot_target & label_target], c="lightblue",marker='X')
        legend += ['target, unprot, true']
        ax.scatter(X_new[target, 0][unprot_target & nlabel_target], X_new[target, 1][unprot_target & nlabel_target], c="thistle",marker='P')
        legend += ['target, unprot, false']

        ax.scatter(X_new[target, 0][prot_target & label_target], X_new[target, 1][prot_target & label_target], c="blue",marker='x')
        legend += ['target, prot, true']
        ax.scatter(X_new[target, 0][prot_target & nlabel_target], X_new[target, 1][prot_target & nlabel_target], c="purple",marker='+')
        legend += ['target, prot, false']

        ax.scatter(X_new[source, 0][unprot_source & label_source], X_new[source, 1][unprot_source & label_source], c="lightgreen", marker='X')
        legend += ['source, unprot, true']
        ax.scatter(X_new[source, 0][unprot_source & nlabel_source], X_new[source, 1][unprot_source & nlabel_source], c="orange", marker='P')
        legend += ['source, unprot, false']

        ax.scatter(X_new[source, 0][prot_source & label_source], X_new[source, 1][prot_source & label_source], c="green", marker='x')
        legend += ['source, prot, true']
        ax.scatter(X_new[source, 0][prot_source & nlabel_source], X_new[source, 1][prot_source & nlabel_source], c="darkorange", marker='+')
        legend += ['source, prot, false']

        ax.legend(legend,loc='lower left')

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    ax = subplots[1]
    ax.set_title('transform (green), target (red)')
    # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    # ax.scatter(X[transform, 0], X[transform, 1], c="g")
    # ax.scatter(X[target, 0], X[target, 1], c="r")
    if is_top_target:
        ax.scatter(X_new[transform, 0][unprot_source], X_new[transform, 1][unprot_source], c="lightgreen", marker='o')
        ax.scatter(X_new[transform, 0][prot_source], X_new[transform, 1][prot_source], c="green", marker='x')
        ax.scatter(X_new[target, 0][unprot_target], X_new[target, 1][unprot_target], c="lightcoral",marker='o')
        ax.scatter(X_new[target, 0][prot_target], X_new[target, 1][prot_target], c="red",marker='x')
        ax.legend(['transform, unprot', 'transform, prot','target, unprot','target, prot'])
    else:

        legend =[]
        ax.scatter(X_new[target, 0][unprot_target & label_target], X_new[target, 1][unprot_target & label_target], c="lightblue",marker='X')
        legend += ['target, unprot, true']
        ax.scatter(X_new[target, 0][unprot_target & nlabel_target], X_new[target, 1][unprot_target & nlabel_target], c="thistle",marker='P')
        legend += ['target, unprot, false']

        ax.scatter(X_new[target, 0][prot_target & label_target], X_new[target, 1][prot_target & label_target], c="blue",marker='x')
        legend += ['target, prot, true']
        ax.scatter(X_new[target, 0][prot_target & nlabel_target], X_new[target, 1][prot_target & nlabel_target], c="purple",marker='+')
        legend += ['target, prot, false']

        ax.scatter(X_new[transform, 0][unprot_source & label_source], X_new[transform, 1][unprot_source & label_source], c="lightgreen", marker='X')
        legend += ['transform, unprot, true']
        ax.scatter(X_new[transform, 0][unprot_source & nlabel_source], X_new[transform, 1][unprot_source & nlabel_source], c="orange", marker='P')
        legend += ['transform, unprot, false']

        ax.scatter(X_new[transform, 0][prot_source & label_source], X_new[transform, 1][prot_source & label_source], c="green", marker='x')
        legend += ['transform, prot, true']
        ax.scatter(X_new[transform, 0][prot_source & nlabel_source], X_new[transform, 1][prot_source & nlabel_source], c="darkorange", marker='+')
        legend += ['transform, prot, false']

        ax.legend(legend,loc='lower left')

        # ax.scatter(X_new[target, 0][unprot_target], X_new[target, 1][unprot_target], c="lightcoral",marker='o')
        # ax.scatter(X_new[target, 0][prot_target], X_new[target, 1][prot_target], c="red",marker='x')
        # ax.scatter(X_new[transform, 0][unprot_source], X_new[transform, 1][unprot_source], c="lightgreen", marker='o')
        # ax.scatter(X_new[transform, 0][prot_source], X_new[transform, 1][prot_source], c="green", marker='x')
        # ax.legend(['target, unprot','target, prot','transform, unprot', 'transform, prot'])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    # ax = subplots[0]
    # # print("perplexities=%d" % (perplexity))
    # ax.set_title('source (blue), target (red)')
    # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
    # # ax.scatter(X_new[green, 0], X_new[green, 1], c="g")
    # # ax.xaxis.set_major_formatter(NullFormatter())
    # # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis("tight")

    # ax = subplots[1]
    # ax.set_title('transform (green), target (red)')
    # # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    # ax.scatter(X_new[transform, 0], X_new[transform, 1], c="g")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="r")
    # # ax.xaxis.set_major_formatter(NullFormatter())
    # # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis("tight")

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()
    
    
def printOT_Diff(source,target,new_source,
    diff_source,diff_target,
    dim_red_method = 'TSNE',
    log=None,title=None,is_top_target=True,save_fig=True,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    
    error = 0
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))

    diff_source = diff_source.clone().detach().cpu().numpy()
    diff_source = np.array([diff_source != 0]).flatten()
    diff_target = diff_target.clone().detach().cpu().numpy()
    diff_target = np.array([diff_target != 0]).flatten()
    ndiff_source = np.logical_not(diff_source)
    ndiff_target = np.logical_not(diff_target)


    # dimenstion reduction
    if dim_red_method == 'TSNE':
        n_components=2
        perplexity=5
        
        log.info('Begin fit TSNE...')
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=10,
            perplexity=100,
            learning_rate="auto",
            n_iter=2000,
        )
        X_new = tsne.fit_transform(X)
        log.info('End fit TSNE...')
        
        
    source = group == 1
    target = group == 2
    transform = group == 3
        
    (fig, subplots) = plt.subplots(1, 2, figsize=(20, 8))

    if title is not None:
        st = fig.suptitle(title, fontsize="xx-large")


    ax = subplots[0]
    # print("perplexities=%d" % (perplexity))
    # ax.set_title('source (blue), target (red)')
    if is_top_target:
        legend =[]
        ax.scatter(X_new[source, 0][diff_source], X_new[source, 1][diff_source], c="green", marker='x')
        legend += ['source, diff=true']
        ax.scatter(X_new[source, 0][ndiff_source], X_new[source, 1][ndiff_source], c="orange", marker='x')
        legend += ['source, diff=false']
        
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, diff=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, diff=false']
        ax.legend(legend,loc='lower left')
    else:
        legend =[]
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, diff=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, diff=false']

        ax.scatter(X_new[source, 0][diff_source], X_new[source, 1][diff_source], c="green", marker='x')
        legend += ['source, diff=true']
        ax.scatter(X_new[source, 0][ndiff_source], X_new[source, 1][ndiff_source], c="orange", marker='x')
        legend += ['source, diff=false']
        ax.legend(legend,loc='lower left')

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    ax = subplots[1]

    if is_top_target:
        legend =[]
        ax.scatter(X_new[transform, 0][diff_source], X_new[transform, 1][diff_source], c="green", marker='x')
        legend += ['transform, diff=true']
        ax.scatter(X_new[transform, 0][ndiff_source], X_new[transform, 1][ndiff_source], c="orange", marker='x')
        legend += ['transform, diff=false']
        
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, diff=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, diff=false']

        ax.legend(legend,loc='lower left')
    else:
        legend =[]
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, diff=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, diff=false']

        ax.scatter(X_new[transform, 0][diff_source], X_new[transform, 1][diff_source], c="green", marker='x')
        legend += ['transform, diff=true']
        ax.scatter(X_new[transform, 0][ndiff_source], X_new[transform, 1][ndiff_source], c="orange", marker='x')
        legend += ['transform, diff=false']


        ax.legend(legend,loc='lower left')

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()
    
def printOT_MultiDiff(source,target,new_source,
    diffs=[],
    dim_red_method = 'TSNE',
    log=None,title=None,is_top_target=True,save_fig=True,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA
    # from matplotlib.ticker import NullFormatter
    # from sklearn import manifold
    
    error = 0
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))


    # dimenstion reduction
    if dim_red_method == 'TSNE':
        n_components=2
        perplexity=5
        
        log.info('Begin fit TSNE...')
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=10,
            perplexity=100,
            learning_rate="auto",
            n_iter=2000,
        )
        X_new = tsne.fit_transform(X)
        log.info('End fit TSNE...')
    elif dim_red_method == 'PCA':
        log.info('Begin fit PCA...')
        pca = PCA(n_components=2)
        X_new =pca.fit_transform(X)
        log.info('End fit PCA...')
        
    diff_n = len(diffs)
    
    (fig, subplots) = plt.subplots(diff_n, 2, figsize=(20, 2+7*diff_n))

    if title is not None:
        st = fig.suptitle(title, fontsize="xx-large")
        
    d_cnt = 0
    for d in diffs:
        diff_source = d['source']
        diff_target = d['target']
        diff_name = d['name']
        
        printDimRedByDiff(X_new,group,d_cnt,fig, subplots,
            diff_source,diff_target,diff_name,
            log=log,is_top_target=is_top_target,save_fig=save_fig,fig_path = fig_path)
        d_cnt+=1
    

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()
    
    
def printDimRedByDiff(X_new,group,d_cnt,fig, subplots,
    diff_source,diff_target,diff_name,
    log=None,is_top_target=True,save_fig=True,fig_path = 'test.png'):
        
        
    source = group == 1
    target = group == 2
    transform = group == 3
    
    diff_source = diff_source.clone().detach().cpu().numpy()
    diff_source = np.array([diff_source != 0]).flatten()
    diff_target = diff_target.clone().detach().cpu().numpy()
    diff_target = np.array([diff_target != 0]).flatten()
    ndiff_source = np.logical_not(diff_source)
    ndiff_target = np.logical_not(diff_target)
    
    
        
#     (fig, subplots) = plt.subplots(1, 2, figsize=(20, 8))

#     if title is not None:
#         st = fig.suptitle(title, fontsize="xx-large")


    ax = subplots[d_cnt,0]
    # print("perplexities=%d" % (perplexity))
    # ax.set_title('source (blue), target (red)')
    if is_top_target:
        legend =[]
        ax.scatter(X_new[source, 0][diff_source], X_new[source, 1][diff_source], c="green", marker='x')
        legend += ['source, '+diff_name+'=true']
        ax.scatter(X_new[source, 0][ndiff_source], X_new[source, 1][ndiff_source], c="orange", marker='x')
        legend += ['source, '+diff_name+'=false']
        
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, '+diff_name+'=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, '+diff_name+'=false']
        ax.legend(legend,loc='lower left')
    else:
        legend =[]
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, '+diff_name+'=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, '+diff_name+'=false']

        ax.scatter(X_new[source, 0][diff_source], X_new[source, 1][diff_source], c="green", marker='x')
        legend += ['source, '+diff_name+'=true']
        ax.scatter(X_new[source, 0][ndiff_source], X_new[source, 1][ndiff_source], c="orange", marker='x')
        legend += ['source, '+diff_name+'=false']
        ax.legend(legend,loc='lower left')

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    ax = subplots[d_cnt,1]

    if is_top_target:
        legend =[]
        ax.scatter(X_new[transform, 0][diff_source], X_new[transform, 1][diff_source], c="green", marker='x')
        legend += ['transform, '+diff_name+'=true']
        ax.scatter(X_new[transform, 0][ndiff_source], X_new[transform, 1][ndiff_source], c="orange", marker='x')
        legend += ['transform, '+diff_name+'=false']
        
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, '+diff_name+'=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, '+diff_name+'=false']

        ax.legend(legend,loc='lower left')
    else:
        legend =[]
        ax.scatter(X_new[target, 0][diff_target], X_new[target, 1][diff_target], c="blue",marker='x')
        legend += ['target, '+diff_name+'=true']
        ax.scatter(X_new[target, 0][ndiff_target], X_new[target, 1][ndiff_target], c="purple",marker='x')
        legend += ['target, '+diff_name+'=false']

        ax.scatter(X_new[transform, 0][diff_source], X_new[transform, 1][diff_source], c="green", marker='x')
        legend += ['transform, '+diff_name+'=true']
        ax.scatter(X_new[transform, 0][ndiff_source], X_new[transform, 1][ndiff_source], c="orange", marker='x')
        legend += ['transform, '+diff_name+'=false']


        ax.legend(legend,loc='lower left')

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

#     if save_fig:
#         log.debug('Saving img to '+ fig_path)
#         plt.savefig(fig_path)
#     else: plt.show()
    
def printOT_2D(source,target,new_source,source_label,target_label,log,save_fig=True,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA
    from matplotlib.ticker import NullFormatter
    # from sklearn import manifold
    
    # print(f'source:\n{source[:5,:]}')
    # print(f'target:\n{target[:5,:]}')

    error = 0
    # print(str(source))s
    # print(str(log))
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)
    # print(f'source:\n{source[:5,:]}')
    # print(f'target:\n{target[:5,:]}')

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    # print(f'X:\n{X[:5,:]}')

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))

    n_components=2
    perplexity=5

    (fig, subplots) = plt.subplots(1, 2, figsize=(15, 5))


    source = group == 1
    target = group == 2
    transform = group == 3

    source_label = source_label.clone().detach().cpu().numpy()
    source_label = np.array([source_label != 0]).flatten()
    target_label = target_label.clone().detach().cpu().numpy()
    target_label = np.array([target_label != 0]).flatten()
    n_source_label = np.logical_not(source_label)
    n_target_label = np.logical_not(target_label)

    # print(f'X[source, 0]:\n{X[source, 0]}')
    # print(f'X[source, 1]:\n{X[source, 1]}')
    legend = []
    ax = subplots[0]
    # print("perplexities=%d" % (perplexity))
    ax.set_title('Before Transformation')
    # ax.scatter(X[source, 0][source_label], X[source, 1][source_label], c="lightblue", marker='x')
    # legend += ['source, label=true']
    # ax.scatter(X[source, 0][n_source_label], X[source, 1][n_source_label], c="blue", marker='o')
    # legend += ['source, label=false']
    # ax.scatter(X[source, 0], X[source, 1], label=prot_source)
    ax.scatter(X[target, 0][target_label], X[target, 1][target_label], c="lightcoral",marker='x')
    legend += ['target, label=true']
    ax.scatter(X[target, 0][n_target_label], X[target, 1][n_target_label], c="red",marker='o')
    legend += ['target, label=false']
    ax.scatter(X[source, 0][source_label], X[source, 1][source_label], c="lightblue", marker='x')
    legend += ['source, label=true']
    ax.scatter(X[source, 0][n_source_label], X[source, 1][n_source_label], c="blue", marker='o')
    legend += ['source, label=false']
    # ax.scatter(X_new[green, 0], X_new[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    
    ax.legend(legend)
    ax.axis("tight")

    legend = []
    ax = subplots[1]
    ax.set_title('After Transformation')
    # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    # ax.scatter(X[transform, 0], X[transform, 1], c="g")
    # ax.scatter(X[target, 0], X[target, 1], c="r")
    # ax.scatter(X[transform, 0][source_label], X[transform, 1][source_label], c="lightgreen", marker='x')
    # legend += ['transformed, label=true']
    # ax.scatter(X[transform, 0][n_source_label], X[transform, 1][n_source_label], c="green", marker='o')
    # legend += ['transformed, label=false']
    ax.scatter(X[target, 0][target_label], X[target, 1][target_label], c="lightcoral",marker='x')
    legend += ['target, label=true']
    ax.scatter(X[target, 0][n_target_label], X[target, 1][n_target_label], c="red",marker='o')
    legend += ['target, label=false']
    ax.scatter(X[transform, 0][source_label], X[transform, 1][source_label], c="lightgreen", marker='x')
    legend += ['transformed, label=true']
    ax.scatter(X[transform, 0][n_source_label], X[transform, 1][n_source_label], c="green", marker='o')
    legend += ['transformed, label=false']
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.legend(legend)
    ax.axis("tight")



    # ax = subplots[0]
    # # print("perplexities=%d" % (perplexity))
    # legend =[]
    # ax.set_title('source (blue), target (red)')
    # ax.scatter(X[source, 0][source_label], X[source, 1][source_label], c="b")
    # legend += ['target, '+diff_name+'=true']
    # ax.scatter(X[target, 0][target_label], X[target, 1][target_label], c="r")
    # # ax.scatter(X_new[green, 0], X_new[green, 1], c="g")
    # # ax.xaxis.set_major_formatter(NullFormatter())
    # # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis("tight")
    # ax.legend(legend,loc='lower left')

    # ax = subplots[1]
    # ax.set_title('transform (green), target (red)')
    # # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    # ax.scatter(X[transform, 0], X[transform, 1], c="g")
    # ax.scatter(X[target, 0], X[target, 1], c="r")
    # # ax.xaxis.set_major_formatter(NullFormatter())
    # # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis("tight")

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()

    
def printOT_2D_prot(source,target,new_source,prot_source,prot_target,title=None,log=None,save_fig=True,fig_path = 'test.png'):

    # should be used rarely so import libraries here instead of header
    import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA
    from matplotlib.ticker import NullFormatter
    # from sklearn import manifold
    
    error = 0
    # print(str(source))s
    print(str(log))
    error, source = cleanInput(source,log)
    error, target = cleanInput(target,log)
    error, new_source = cleanInput(new_source,log)

    log.info('Printing OT...')
    X = source.clone().detach().cpu().numpy()
    group = np.ones(X.shape[0])

    n = target.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*2
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))

    n = new_source.clone().detach().cpu().numpy()
    n_group = np.ones(n.shape[0])*3
    X = np.concatenate((X,n),axis=0)
    group = np.concatenate((group,n_group))
    
    unprot_source = np.logical_not(prot_source)
    unprot_target = np.logical_not(prot_target)

    # log.debug('X shape: ' + str(X.shape))
    # log.debug('group shape: ' + str(group.shape))

    n_components=2
    perplexity=5

    (fig, subplots) = plt.subplots(1, 2, figsize=(15, 5))
    if title is not None:
        st = fig.suptitle(title, fontsize="x-large")


    source = group == 1
    target = group == 2
    transform = group == 3


    ax = subplots[0]
    # print("perplexities=%d" % (perplexity))
    ax.set_title('source (blue), target (red)')
    ax.scatter(X[source, 0][unprot_source], X[source, 1][unprot_source], c="lightblue", marker='x')
    ax.scatter(X[source, 0][prot_source], X[source, 1][prot_source], c="blue", marker='o')
    # ax.scatter(X[source, 0], X[source, 1], label=prot_source)
    ax.scatter(X[target, 0][unprot_target], X[target, 1][unprot_target], c="lightcoral",marker='x')
    ax.scatter(X[target, 0][prot_target], X[target, 1][prot_target], c="red",marker='o')
    # ax.scatter(X_new[green, 0], X_new[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    
    ax.legend(["one" , "two",'three','four'])
    ax.axis("tight")

    ax = subplots[1]
    ax.set_title('transform (green), target (red)')
    # ax.scatter(X_new[source, 0], X_new[source, 1], c="b")
    # ax.scatter(X_new[target, 0], X_new[target, 1], c="g")
    # ax.scatter(X[transform, 0], X[transform, 1], c="g")
    # ax.scatter(X[target, 0], X[target, 1], c="r")
    ax.scatter(X[transform, 0][unprot_source], X[transform, 1][unprot_source], c="lightgreen", marker='x')
    ax.scatter(X[transform, 0][prot_source], X[transform, 1][prot_source], c="green", marker='o')
    ax.scatter(X[target, 0][unprot_target], X[target, 1][unprot_target], c="lightcoral",marker='x')
    ax.scatter(X[target, 0][prot_target], X[target, 1][prot_target], c="red",marker='o')
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    if save_fig:
        log.debug('Saving img to '+ fig_path)
        plt.savefig(fig_path)
    else: plt.show()
    
    
def cleanInput(input_array,log):
    error = 0 # error flag to indicate if data is in good condition to proceed
    if input_array is None:
        msg = 'Input is None'
        if log is None: print('WARNING: '+msg)
        else: log.error(msg)
    elif type(input_array) is np.ndarray:
        msg = 'Input is Numpy, converting to tensors....'
        if log is None: print('WARNING: '+msg)
        else: log.debug(msg)
        input_array = torch.from_numpy(input_array).to('cpu').float()
    # print(type(input_array))

    if torch.isnan(input_array).any():
        msg = 'Nan in Input'
        if log is None: print('WARNING: '+msg)
        else: log.error(msg)
        error = 1

    return(error, input_array)