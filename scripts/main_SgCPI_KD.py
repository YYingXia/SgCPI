import os
import random
import pandas as pd
import torch
import torch.optim as optim
from torchnet import meter
import numpy as np
import shutil
import sys
import time
import joblib
from valid_metrices import eval_metrics, th_eval_metrics
import warnings
import pickle
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from tqdm import tqdm, trange
import gc
from torch_geometric.data import Batch
import torch.nn.functional as F
import math
from torch_geometric.nn.inits import glorot, ones, reset

warnings.filterwarnings('ignore')

class Config():
    def __init__(self):
        self.context_num = 10
        self.sim_top_k = 100

        self.lamb_sup = 1
        self.lamb_emb = 1
        self.lamb_link = 0
        self.lamb_node_emb = 0
        self.lamb_pair_emb = 0

        self.hidden_size = 256

        # optimize
        self.train_batch_size = 64
        self.batch_size = 512
        self.lr = 0.0001
        self.epochs = 200
        self.runseed = 0
        self.dropratio = 0.1
        self.l2_weight = 0.00001
        self.es_loss_delta = 0.0001

        localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.checkpoint_path = './checkpoints/{}'.format(localtime)
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)


    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))


class load_SgCPI_KD_dataset(Dataset):
    def __init__(self, fold, context_num, kd_target=False, teacher_train_label=None, train_prot_emb=None, train_compound_emb=None,
                 teacher_train_edge_label=None):
        self.kd_target = kd_target
        self.fold = fold
        self.protein_msa_num = 64
        self.subset = None
        self.teacher_train_edge_label = teacher_train_edge_label
        self.context_num = context_num

        self.protein_list = []
        with open('../dataset/protein.fasta','r') as f:
            text = f.readlines()
        for i in range(0,len(text),2):
            self.protein_list.append(text[i].strip()[1:])

        self.compound_list = pd.read_csv('../dataset/compound_smiles.csv')['cid'].tolist()
        self.compound_list = [str(cid) for cid in self.compound_list]

        print('load protein fea...')
        self.protein_fea = self.cal_protein_meanpool_fea(self.protein_list)
        self.protein_emb_dim = self.protein_fea.shape[-1]

        print(self.protein_fea.shape)

        print('load compound fea...')
        self.compound_fea = self.cal_compound_grover_fea()
        self.compound_emb_dim = self.compound_fea.shape[1]

        if teacher_train_label is not None:
            self.teacher_train_label = teacher_train_label.tolist()
        if train_prot_emb is not None:
            self.train_prot_emb = torch.tensor(train_prot_emb)
        if train_compound_emb is not None:
            self.train_compound_emb = torch.tensor(train_compound_emb)

        self.protein_id_dict = dict(zip(list(range(len(self.protein_list))), self.protein_list))
        compound_list2 = [int(cid) for cid in self.compound_list]
        self.compound_id_dict = dict(zip(list(range(len(compound_list2))), compound_list2))


    def __getitem__(self, index):
        if self.subset == 'train':
            context = torch.tensor(self.anchor_context[index])
            compound_id = context[:,0].long()
            prot_id = context[:,1].long()
            label = context[:,2].float()
            teacher_label = context[:,3].float()
            train_idxs = context[:,4].long()
            
            prot_fea = self.protein_fea[prot_id]
            compound_fea = self.compound_fea[compound_id]

            tea_prot_emb = self.train_prot_emb[train_idxs,:]
            tea_compound_emb = self.train_compound_emb[train_idxs,:]

            return prot_fea, compound_fea, label, teacher_label, tea_prot_emb, tea_compound_emb

        else:
            compound_id = self.pair_index[index][0]
            prot_id = self.pair_index[index][1]
            label = self.pair_label[index]
            label = torch.tensor(label).long()
            prot_fea = self.protein_fea[prot_id]
            compound_fea = self.compound_fea[compound_id]
            return prot_fea, compound_fea, label

    def __len__(self):
        if self.subset == 'train':
            return len(self.anchor_context)
        else:
            return len(self.pair_index)

    def cal_protein_meanpool_fea(self, protein_list):
        prot_fea_dict = joblib.load('../dataset/protein_fea_dict.pt')
        prot_fea = []
        for prot_id in protein_list:
            prot_fea.append(prot_fea_dict[prot_id][:-1].unsqueeze(0))
        prot_fea = torch.cat(prot_fea, dim=0)

        del prot_fea_dict
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()

        return prot_fea

    def cal_compound_grover_fea(self):
        compound_fea_dict = joblib.load('../dataset/compound_fea_dict.pt')
        compound_fea = []
        for compound_id in self.compound_list:
            compound_fea.append(compound_fea_dict[str(compound_id)].unsqueeze(0))
        compound_fea = torch.cat(compound_fea, dim=0)

        del compound_fea_dict
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()

        return compound_fea

    def load_compound_protein_edge(self, subset):
        self.subset = subset
        protein_id_dict = dict(zip(self.protein_list, list(range(len(self.protein_list)))))
        compound_id_dict = dict(zip(self.compound_list, list(range(len(self.compound_list)))))

        index_list = []
        label_list = []

        with open('../dataset/inductive/fold{}/{}.txt'.format(self.fold, subset),'r') as f:
            text = f.readlines()[1:]

        for line in text:
            cid = line.split('\t')[0]
            uniport = line.split('\t')[2]
            label = int(line.split('\t')[4].strip())
            index_list.append([compound_id_dict[cid], protein_id_dict[uniport], cid, uniport])
            label_list.append(label)

        self.pair_index = index_list
        self.pair_label = label_list

        print('edge_num: ',len(self.pair_index))
        return

    def load_train_edge_context(self):
        self.subset = 'train'

        protein_id_dict = dict(zip(self.protein_list, list(range(len(self.protein_list)))))
        compound_id_dict = dict(zip(self.compound_list, list(range(len(self.compound_list)))))

        teacher_label_list = torch.tensor(self.teacher_train_label)
        teacher_label_list = teacher_label_list.tolist()

        data_list = []

        with open('../dataset/inductive/fold{}/train.txt'.format(self.fold), 'r') as f:
            text = f.readlines()[1:]

        for i, line in enumerate(text):
            cid = line.split('\t')[0]
            uniport = line.split('\t')[2]
            label = int(line.split('\t')[4].strip())
            teacher_label = teacher_label_list[i]
            data_list.append([compound_id_dict[cid], protein_id_dict[uniport], label, teacher_label, i])

        print('edge_num: ',len(data_list))

        data_list = np.array(data_list)

        label_list = data_list[:, 2]

        if not torch.tensor(label_list).float().equal(self.teacher_train_edge_label):
            raise ValueError

        anchor_protein_compound_context = []
        for i in range(len(self.protein_list)):
            context = data_list[data_list[:,1]==i]
            np.random.shuffle(context)
            context_len = context.shape[0]
            if context_len >= self.context_num:
                for j in range(0, context_len, self.context_num):
                    if j + self.context_num <= context_len:
                        subcontext = context[j: j + self.context_num]
                    else:
                        subcontext = np.concatenate([context[j:], context[:self.context_num - (context_len-j)]], axis=0)
                    assert subcontext.shape[0] == self.context_num
                    anchor_protein_compound_context.append(subcontext)
            elif context_len > 0:
                subcontext = context
                while True:
                    subcontext = np.concatenate([subcontext, subcontext], axis=0)
                    if len(subcontext) >= self.context_num:
                        break
                subcontext = subcontext[:self.context_num]
                anchor_protein_compound_context.append(subcontext)

        anchor_compound_protein_context = []

        self.anchor_context = anchor_protein_compound_context + anchor_compound_protein_context
        random.shuffle(self.anchor_context)

        return



class PairModel_simprot_encoder_clf_AE(torch.nn.Module):
    def __init__(self, prot_encoder, compound_encoder, hidden_channels, dropratio, sim_top_k, train_prot_avg_emb, train_compound_avg_emb):
        super(PairModel_simprot_encoder_clf_AE,self).__init__()

        self.prot_encoder = prot_encoder
        self.compound_encoder = compound_encoder

        if sim_top_k is None:
            self.sim_top_k = train_prot_avg_emb.shape[0]
        else:
            self.sim_top_k = sim_top_k

        self.hidden_channels = hidden_channels

        self.train_prot_avg_emb = torch.autograd.Variable(train_prot_avg_emb, requires_grad=False) 
        self.train_compound_avg_emb = torch.autograd.Variable(train_compound_avg_emb, requires_grad=False) 

        print('--AE--')
        print('sim top k : ', self.sim_top_k)
        print(self.train_prot_avg_emb.shape)
        print(self.train_compound_avg_emb.shape)

        self.emb_part_num = 10000

        self.train_prot_avg_emb_list = []
        train_prot_num = self.train_prot_avg_emb.shape[0] // self.emb_part_num
        if train_prot_num == 0:
            self.train_prot_avg_emb_list.append(self.train_prot_avg_emb)
        else:
            for ii in range(train_prot_num):
                if ii == train_prot_num - 1:
                    self.train_prot_avg_emb_list.append(
                        self.train_prot_avg_emb[ii * self.emb_part_num: self.train_prot_avg_emb.shape[0]])
                else:
                    self.train_prot_avg_emb_list.append(
                        self.train_prot_avg_emb[ii * self.emb_part_num: (ii + 1) * self.emb_part_num])

        self.train_compound_avg_emb_list = []
        train_compound_num = self.train_compound_avg_emb.shape[0]//self.emb_part_num
        if train_compound_num == 0:
            self.train_compound_avg_emb_list.append(self.train_compound_avg_emb)
        else:
            for ii in range(train_compound_num):
                if ii == train_compound_num -1:
                    self.train_compound_avg_emb_list.append(self.train_compound_avg_emb[ii*self.emb_part_num: self.train_compound_avg_emb.shape[0]])
                    break
                else:
                    self.train_compound_avg_emb_list.append(self.train_compound_avg_emb[ii*self.emb_part_num: (ii+1)*self.emb_part_num])


        self.p_q_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))
        self.m_q_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))

        self.p_k_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))
        self.m_k_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))

        self.p_v_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))
        self.m_v_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels))


        self.p_out_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels*2, hidden_channels), torch.nn.ReLU(inplace=True), torch.nn.Linear(hidden_channels, hidden_channels))
        self.m_out_lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels*2, hidden_channels), torch.nn.ReLU(inplace=True), torch.nn.Linear(hidden_channels, hidden_channels))


        self.pp_k_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.mm_k_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.mp_k_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.pm_k_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.pp_v_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.mm_v_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.mp_v_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.pm_v_rel = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))


        self.clf = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.pp_k_rel)
        glorot(self.mm_k_rel)
        glorot(self.mp_k_rel)
        glorot(self.pm_k_rel)

        glorot(self.pp_v_rel)
        glorot(self.mm_v_rel)
        glorot(self.mp_v_rel)
        glorot(self.pm_v_rel)


    def forward(self, prot_data, compound_data, teacher_prot_index=None, emb=False):
        prot_emb = self.prot_encoder(prot_data)
        compound_emb = self.compound_encoder(compound_data)

        prot_emb_q = self.p_q_lin(prot_emb)
        compound_emb_q = self.m_q_lin(compound_emb)

        train_prot_list_k = [self.p_k_lin(x) for x in self.train_prot_avg_emb_list]
        train_compound_list_k = [self.m_k_lin(x) for x in self.train_compound_avg_emb_list]

        train_prot_v = torch.cat([self.p_v_lin(x) for x in self.train_prot_avg_emb_list], dim=0)
        train_compound_v = torch.cat([self.m_v_lin(x) for x in self.train_compound_avg_emb_list], dim=0)

        # prot_emb_q = prot_emb
        # compound_emb_q = compound_emb

        # train_prot_list_k = self.train_prot_avg_emb_list
        # train_prot_v = self.train_prot_avg_emb
        #
        # train_compound_list_k = self.train_compound_avg_emb_list
        # train_compound_v = self.train_compound_avg_emb

        # sim_pp_emb, topk_pp_emb = self.topksim(prot_emb_q, train_prot_list_k, train_prot_v, self.sim_top_k, self.mp_k_rel, self.mp_v_rel)
        sim_mp_emb, topk_mp_emb = self.topksim(compound_emb_q, train_prot_list_k, train_prot_v, self.sim_top_k, self.mp_k_rel, self.mp_v_rel)
        # sim_mp_emb, topk_mp_emb = self.randomsim(compound_emb_q, train_prot_list_k, train_prot_v, self.sim_top_k, self.mp_k_rel, self.mp_v_rel)

        # sim_mm_emb, topk_mm_emb = self.topksim(compound_emb_q, train_compound_list_k, train_compound_v, self.sim_top_k, self.pm_k_rel, self.pm_v_rel)
        sim_pm_emb, topk_pm_emb = self.topksim(prot_emb_q, train_compound_list_k, train_compound_v, self.sim_top_k, self.pm_k_rel, self.pm_v_rel)
        # sim_pm_emb, topk_pm_emb = self.randomsim(prot_emb_q, train_compound_list_k, train_compound_v, self.sim_top_k, self.pm_k_rel, self.pm_v_rel)

        # sim_mm_emb = self.attn_sim(compound_emb_q, topk_pm_emb, topk_pm_emb)
        # sim_pp_emb = self.attn_sim(prot_emb_q, topk_mp_emb, topk_mp_emb)

        prot_emb_2 = torch.cat([prot_emb, sim_pm_emb], dim=-1)
        compound_emb_2 = torch.cat([compound_emb, sim_mp_emb], dim=-1)

        prot_emb_2 = self.p_out_lin(prot_emb_2)
        compound_emb_2 = self.m_out_lin(compound_emb_2)

        prot_emb_2_norm = F.normalize(prot_emb_2)
        compound_emb_2_norm = F.normalize(compound_emb_2)

        adj_pred = (torch.sum(prot_emb_2_norm * compound_emb_2_norm, dim=1) + 1) * 0.5


        # prot_emb_2 = self.p_out_lin(prot_emb_2)
        # compound_emb_2 = self.m_out_lin(compound_emb_2)
        #
        # pair_emb = torch.cat([prot_emb_2, compound_emb_2], dim=-1)
        #
        # adj_pred = self.clf(pair_emb).squeeze(1)
        # adj_pred = F.sigmoid(adj_pred)


        if emb:
            return adj_pred, prot_emb_2, compound_emb_2
        else:
            return adj_pred



    def topksim(self, query, dataset_list_k, dataset_v, sim_top_k, k_rel, v_rel):
        topk_list = []
        topkindices_list = []

        # query = query @ k_rel
        # dataset_list_k = [emb @ k_rel for emb in dataset_list_k]
        for i, emb in enumerate(dataset_list_k):
            sim_score = query @ emb.T
            topk, topkindices = torch.topk(sim_score, sim_top_k)
            topk_list.append(topk)
            topkindices_list.append(topkindices + self.emb_part_num * i)

        topk_list = torch.cat(topk_list, dim=1)
        topkindices_list = torch.cat(topkindices_list, dim=1)

        topk, topkindices2 = torch.topk(topk_list, sim_top_k)
        topkindices = torch.gather(topkindices_list, dim=1, index=topkindices2)

        topk = topk / math.sqrt(self.hidden_channels)
        topk_softmax = torch.softmax(topk, dim=1)  # B * k
        topk_softmax = topk_softmax.unsqueeze(1)

        topk_emb = dataset_v[topkindices, :]  # B * k * D

        # sim_emb = sim_emb @ v_rel

        sim_emb = topk_softmax @ topk_emb
        sim_emb = sim_emb.squeeze(1)

        return sim_emb, topk_emb

    def randomsim(self, query, dataset_list_k, dataset_v, sim_top_k, k_rel, v_rel):
        topk_list = []
        topkindices_list = []

        query = torch.rand(query.shape, device=query.device)
        # query = query @ k_rel
        # dataset_list_k = [emb @ k_rel for emb in dataset_list_k]
        for i, emb in enumerate(dataset_list_k):
            sim_score = query @ emb.T
            topk, topkindices = torch.topk(sim_score, sim_top_k)
            topk_list.append(topk)
            topkindices_list.append(topkindices + self.emb_part_num * i)

        topk_list = torch.cat(topk_list, dim=1)
        topkindices_list = torch.cat(topkindices_list, dim=1)

        topk, topkindices2 = torch.topk(topk_list, sim_top_k)
        topkindices = torch.gather(topkindices_list, dim=1, index=topkindices2)

        topk = topk / math.sqrt(self.hidden_channels)
        topk_softmax = torch.softmax(topk, dim=1)  # B * k
        topk_softmax = topk_softmax.unsqueeze(1)

        topk_emb = dataset_v[topkindices, :]  # B * k * D

        # sim_emb = sim_emb @ v_rel

        sim_emb = topk_softmax @ topk_emb
        sim_emb = sim_emb.squeeze(1)

        return sim_emb, topk_emb


    def randomsim2(self, query, dataset_list_k, dataset_v, sim_top_k, k_rel, v_rel):
        sim_emb_list = []
        for dataset_k in dataset_list_k:
            dataset_k = dataset_k.unsqueeze(0).repeat(query.shape[0], 1, 1)
            weights = torch.ones(dataset_k.shape[:2])
            index = torch.multinomial(weights, sim_top_k)
            index = index.unsqueeze(-1).repeat(1, 1, dataset_k.shape[2]).to(dataset_k.device)
            sim_emb = torch.gather(dataset_k, 1, index)
            sim_emb_list.append(sim_emb)

        dataset_k = torch.cat(sim_emb_list, dim=1)
        weights = torch.ones(dataset_k.shape[:2])
        index = torch.multinomial(weights, sim_top_k)
        index = index.unsqueeze(-1).repeat(1, 1, dataset_k.shape[2]).to(dataset_k.device)
        sim_emb = torch.gather(dataset_k, 1, index)
        sim_emb = torch.mean(sim_emb, dim=1)
        return sim_emb


    def attn_sim(self, query, dataset_k, dataset_v):
        query = query.unsqueeze(1)  # B * 1 * D
        sim_score = query @ dataset_k.transpose(1,2)  # B * 1 * K
        sim_score = sim_score / math.sqrt(self.hidden_channels)

        sim_softmax = torch.softmax(sim_score, dim=-1)

        sim_emb = sim_softmax @ dataset_v
        sim_emb = sim_emb.squeeze(1)

        return sim_emb

class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, dropratio):
        super(MLP,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels))

    def forward(self, x):
        x = self.mlp(x)
        return x

class ProtMLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, dropratio):
        super(ProtMLP,self).__init__()
        self.esm_mlp = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels))

        self.grasr_mlp = torch.nn.Sequential(
            torch.nn.Linear(400, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels))

    def forward(self, x):
        esm_fea = x[:, :768]
        grasa_fea = x[:, 768:]
        esm_fea = self.esm_mlp(esm_fea)
        grasa_fea = self.grasr_mlp(grasa_fea)
        x = torch.cat([esm_fea, grasa_fea], dim=1)
        x = self.mlp(x)
        return x

def valid(opt, model, device, valid_data):
    Numres = valid_data.__len__()
    valid_batchsize = opt.batch_size
    while True:
        if (Numres - 1) % valid_batchsize == 0:
            valid_batchsize += 1
        else:
            break
    dataloader = DataLoader(valid_data, batch_size=valid_batchsize, shuffle=False, drop_last=False, pin_memory=True)
    model.to(device)
    model.eval()
    score_list = []
    target_list = []
    with torch.no_grad():
        for ii, (prot_data, compound_data, label) in enumerate(dataloader):

            compound_data = compound_data.to(device)
            prot_data = prot_data.to(device)
            label = label.to(device)

            score = model(prot_data, compound_data)
            # score = torch.sigmoid(score)
            score_list.extend(score.detach().cpu().tolist())
            target_list.extend(label.detach().cpu().tolist())
    score_list = np.array(score_list)
    target_list = np.array(target_list)
    return score_list, target_list

def train(opt, fold):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(opt.runseed)
    np.random.seed(opt.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.runseed)

    teacher_train_label_path = '../dataset/teacher_emb/train_fold{}.pkl'.format(fold)
    print('load ', teacher_train_label_path)
    with open(teacher_train_label_path, 'rb') as f:
        [train_log_pred, train_edge_label, train_prot_emb, train_compound_emb, train_pair_emb,
         train_prot_id_list, train_prot_emb_list, train_compound_id_list, train_compound_emb_list] = pickle.load(f)

    train_prot_avg_emb = torch.tensor(train_prot_emb_list).to(device)
    train_compound_avg_emb = torch.tensor(train_compound_emb_list).to(device)


    loaded_data = load_SgCPI_KD_dataset(fold=fold, context_num=opt.context_num, kd_target=True,
                                        teacher_train_label=train_log_pred, train_prot_emb=train_prot_emb, train_compound_emb=train_compound_emb,
                                        teacher_train_edge_label=train_edge_label)

    prot_encoder = ProtMLP(input_channels=loaded_data.protein_emb_dim, hidden_channels=opt.hidden_size, dropratio=opt.dropratio)
    compound_encoder = MLP(input_channels=loaded_data.compound_emb_dim, hidden_channels=opt.hidden_size, dropratio=opt.dropratio)

    model = PairModel_simprot_encoder_clf_AE(prot_encoder=prot_encoder, compound_encoder=compound_encoder, hidden_channels=opt.hidden_size,
                                          dropratio=opt.dropratio, sim_top_k=opt.sim_top_k,
                                          train_prot_avg_emb=train_prot_avg_emb, train_compound_avg_emb=train_compound_avg_emb).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2_weight)

    criterion_l = torch.nn.BCELoss(reduction="mean").to(device)
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean").to(device)
    criterion_f = torch.nn.MSELoss(reduction="mean").to(device)

    best_db_valid_auc = 0
    es = 0
    for epoch in range(0, opt.epochs + 1):
        loaded_data.load_train_edge_context()

        train_loader = DataLoader(loaded_data, batch_size=opt.train_batch_size, shuffle=True, drop_last=True, pin_memory=True)

        model.train()
        loss_meter = meter.AverageValueMeter()
        loss_sup_meter = meter.AverageValueMeter()
        loss_emb_meter = meter.AverageValueMeter()
        for ii, (prot_data, compound_data, label, teacher_label, teacher_protein_emb, teacher_compound_emb) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            compound_data = compound_data.to(device)
            prot_data = prot_data.to(device)
            label = label.to(device)
            teacher_label = teacher_label.to(device)

            teacher_protein_emb = teacher_protein_emb.to(device).reshape(-1, teacher_protein_emb.shape[-1])
            teacher_compound_emb = teacher_compound_emb.to(device).reshape(-1, teacher_compound_emb.shape[-1])


            compound_data = compound_data.reshape(-1, compound_data.shape[-1])
            prot_data = prot_data.reshape(-1, prot_data.shape[-1])
            label = label.reshape(-1)
            teacher_label = teacher_label.reshape(-1)

            train_pred, prot_emb, compound_emb = model(prot_data, compound_data, emb=True)

            loss_sup = criterion_l(train_pred, label)

            loss_emb = criterion_f(prot_emb, teacher_protein_emb)
            loss_emb += criterion_f(compound_emb, teacher_compound_emb)

            train_pred = train_pred.reshape(-1, opt.context_num)
            teacher_label = teacher_label.reshape(-1, opt.context_num)

            train_pred = torch.log_softmax(train_pred, dim=1)
            teacher_label = torch.softmax(teacher_label, dim=1)

            loss = opt.lamb_sup * loss_sup + opt.lamb_emb * 1
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            loss_sup_meter.add(loss_sup.item())
            loss_emb_meter.add(loss_emb.item())

        print('|| Epoch {} | train loss={:.5f} sup_loss={:.5f} emb_loss={:.5f}||'.format(
                        epoch, loss_meter.mean, loss_sup_meter.mean, loss_emb_meter.mean))

        loss_meter.reset()
        loss_sup_meter.reset()
        loss_emb_meter.reset()

        loaded_data.load_compound_protein_edge('valid')
        valid_pred, valid_target = valid(opt, model, device, loaded_data)
        dbval_th, dbval_rec, dbval_pre, dbval_F1, dbval_spe, dbval_mcc, dbval_auc, dbval_ap, re0_5, re1, re2, re5, pred_class = eval_metrics(valid_pred, valid_target)
        print('valid result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
              .format(dbval_th, dbval_rec, dbval_pre, dbval_F1, dbval_spe, dbval_mcc, dbval_auc, dbval_ap))

        loaded_data.load_compound_protein_edge('test')
        test_pred, test_target = valid(opt, model, device, loaded_data)
        dbtest_th,dbtest_rec, dbtest_pre, dbtest_F1, dbtest_spe, dbtest_mcc, dbtest_auc, dbtest_ap, re0_5, re1, re2, re5, pred_class = th_eval_metrics(dbval_th, test_pred, test_target)
        print('test  result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
              .format(dbtest_th,dbtest_rec, dbtest_pre, dbtest_F1, dbtest_spe, dbtest_mcc, dbtest_auc, dbtest_ap))

        if dbval_auc > best_db_valid_auc:
            es = 0
            save_path = '{}/model_epoch{}.pth'.format(opt.checkpoint_path, fold, epoch)
            print('save net: ', save_path)
            torch.save(model.state_dict(), save_path)

            best_db_valid_auc = dbval_auc
            best_dbvalid_results = [dbval_th, dbval_rec, dbval_pre, dbval_F1, dbval_spe, dbval_mcc, dbval_auc, dbval_ap]
            best_dbtest_results = [dbtest_th, dbtest_rec, dbtest_pre, dbtest_F1, dbtest_spe, dbtest_mcc, dbtest_auc, dbtest_ap]
            print('## EPOCH {} || higher AUC of db valid'.format(epoch))
        else:
            es += 1
            
        if es == 10:
            break

    print('===================Result==================='.format(fold))
    print('valid result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
          .format(best_dbvalid_results[0], best_dbvalid_results[1], best_dbvalid_results[2], best_dbvalid_results[3], best_dbvalid_results[4], best_dbvalid_results[5], best_dbvalid_results[6], best_dbvalid_results[7]))
    print('result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
          .format(best_dbtest_results[0], best_dbtest_results[1], best_dbtest_results[2], best_dbtest_results[3], best_dbtest_results[4], best_dbtest_results[5], best_dbtest_results[6], best_dbtest_results[7]))

    return best_dbvalid_results, best_dbtest_results

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--fold", dest="fold", help="choose from 0 to 4.")
    return parser.parse_args()


if __name__ == "__main__":
    opt = Config()
    opt.print_config()

    args = parse_args()
    fold = args.fold

    train(opt, fold)

