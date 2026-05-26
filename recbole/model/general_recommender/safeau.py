
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class SaFeAU(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(SaFeAU, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.gamma1 = config['gamma1']
        self.top_k = config['top_k']
        self.K = config['K']
        self.gamma2 = config['gamma2']
        self.train_strategy = config['train_strategy']
        self.train_stage = 'pretrain'
        if self.train_strategy == 'MF':
            self.use_mf = True
        elif self.train_strategy == 'GODE':
            self.use_mf = False
        else:
            self.use_mf = None
        
        # define layers and loss
        self.t = config['t']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.encoder = Encoder(self.use_mf, self.n_users, self.n_items, self.embedding_size, self.norm_adj, t = torch.tensor([0, self.t]))
        
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        # parameters initialization
        self.apply(xavier_normal_initialization)
    
    def ft_init(self):
        self.use_mf = False
        self.train_stage = 'finetuning'
        self.encoder.update(use_mf=self.use_mf)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
    
    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key,value in data_dict.items():
            A[key] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    
    def forward(self, user, item):
        if self.train_strategy == 'MF_init' and self.train_stage == 'pretrain':
            self.encoder.update(use_mf = self.training)
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
    
    def DynamicRouting(self, node_embeddings, num_interest, pos_embeddings, epsilon=1e-3, max_r= 4, compute_intra_loss = True):
        K_interest = num_interest
        node_num, _ = node_embeddings.shape
        capsule_weight = torch.randn(K_interest, node_num, device=self.device)
        node_embeddings_t = node_embeddings.T
        prev_capsule_weight = capsule_weight.clone()
        
        for k in range(max_r):
            capsule_softmax_weight = F.softmax(capsule_weight, dim=1)
            interest_capsule = torch.matmul(capsule_softmax_weight, node_embeddings)
            interest_capsule = F.normalize(interest_capsule, p=2, dim=1)
            if k < max_r - 1:
                delta_weight = torch.matmul(interest_capsule, node_embeddings_t)
                capsule_weight = capsule_weight + delta_weight
                relative_diff = torch.abs(capsule_weight - prev_capsule_weight).mean() / (
                    torch.abs(prev_capsule_weight).mean() + 1e-9)
                if relative_diff < epsilon:
                    break
                prev_capsule_weight = capsule_weight.clone()
        
        capsule_probs = F.softmax(capsule_weight, dim=1)  # [K, N]
        
        top_k_values, top_k_indices = torch.topk(capsule_probs.T, k = self.top_k, dim=1)  # [N, top_k]
        
        multi_intra_loss = 0.0
        if compute_intra_loss:
            multi_intra_loss = self.compute_multi_capsule_contrastive_loss(node_embeddings, top_k_indices, top_k_values, pos_embeddings)
        return multi_intra_loss
    
    def compute_multi_capsule_contrastive_loss(self, node_embeddings, top_k_indices, top_k_values, pos_embeddings, sample_ratio=0.01):
        node_num = node_embeddings.shape[0]
        sample_size = max(2, int(node_num * sample_ratio))
        sampled_indices = torch.randint(0, node_num, (sample_size,), device=self.device)
        sampled_embeddings = node_embeddings[sampled_indices]
        
        total_loss = torch.tensor(0.0, device=self.device)
        valid_nodes = 0
        sampled_top_k_indices = top_k_indices[sampled_indices]
        sampled_top_k_values = top_k_values[sampled_indices]
        
        for i in range(sample_size):
            original_idx = sampled_indices[i]

            node_capsules = sampled_top_k_indices[i]
            node_weights = sampled_top_k_values[i]
            node_loss = torch.tensor(0.0, device=self.device)
            
            for cap_idx, weight in zip(node_capsules, node_weights):
                same_capsule_mask = (top_k_indices == cap_idx).any(dim=1)
                same_capsule_mask[original_idx] = False
                
                same_capsule_count = same_capsule_mask.sum()
                if same_capsule_count == 0:
                    continue 
                
                same_capsule_indices = torch.where(same_capsule_mask)[0]
                node_i_expanded = sampled_embeddings[i].unsqueeze(0).expand(same_capsule_count, -1)
                
                pos_embeddings_selected = pos_embeddings[same_capsule_indices]
                
                alignment_losses = self.alignment(node_i_expanded, pos_embeddings_selected)
                weighted_alignment = alignment_losses.mean()
                alignment_contribution = weight * weighted_alignment
                node_loss += alignment_contribution
                
            if node_loss > 0:
                total_loss += node_loss 
                valid_nodes += 1
                
        return total_loss / valid_nodes if valid_nodes > 0 else torch.tensor(0.0, device=self.device)
    
    
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        
        align = self.alignment(user_e, item_e)
        uniform = self.gamma1 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        
        item_intra_loss = self.DynamicRouting(item_e, self.K, user_e, epsilon = 1e-3)
        intra_loss = self.gamma2 * (item_intra_loss)

        return align + uniform + intra_loss
    
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.encoder.user_embedding(user)
        item_e = self.encoder.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            if self.train_strategy == 'MF_init':
                self.encoder.update(use_mf=self.training)
            self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
        user_e = self.restore_user_e[user]
        all_item_e = self.restore_item_e

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

class Encoder(nn.Module):
   
    def __init__(self, use_mf, n_users, n_items, emb_size, norm_adj, t=1.0, solver='euler'):
        super(Encoder, self).__init__()
        self.use_mf = use_mf
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.t = t
        self.solver = solver
        self.e = None
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.item_embedding = nn.Embedding(n_items, emb_size)
    
    def update(self, use_mf):
        self.use_mf = use_mf

    def ode_func(self, t,x):
        return torch.spmm(self.norm_adj, x) + self.e
    

    def get_all_embeddings(self):
        e = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        if not self.use_mf:
            self.e = e
            t = self.t.type_as(e)
            e = odeint(self.ode_func, e, t,method=self.solver)[1]
        return torch.split(e, [self.n_users, self.n_items])
    
    def forward(self, user_id, item_id):
        user_e, item_e = self.get_all_embeddings()
        return user_e[user_id], item_e[item_id]