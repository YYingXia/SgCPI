import os
import torch
import torch.optim as optim
from torchnet import meter
import numpy as np
import time
import joblib
import argparse
from valid_metrices import eval_metrics, th_eval_metrics
import warnings
import torch_geometric.transforms as T
from data_loader import SubgraphLinkNeighborLoader
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_zip, Batch
from HGNN import HGNN
warnings.filterwarnings('ignore')

class Config():
    def __init__(self):
        self.auxedge = False
        self.root = '../dataset/sgcpi_data_process'

        self.hidden_size = 256
        self.n_layers = 2

        # dataloader
        num_neighbors = dict()
        num_neighbors['compound','active','protein'] = [10,10]
        num_neighbors['compound','inactive','protein'] = [10,10]
        num_neighbors['compound','pred','protein'] = [0,0]
        num_neighbors['protein','rev_active','compound'] = [10,10]
        num_neighbors['protein','rev_inactive','compound'] = [10,10]
        num_neighbors['protein','rev_pred','compound'] = [0,0]
        self.num_neighbors = num_neighbors

        self.subgraph_neighbors = [(10, 10, 0, 2)]
        self.compgraph_neighbors = [(10, 10, 0, 1000)]
        self.num_subgraph = len(self.subgraph_neighbors)

        self.dropout_adj_p = 0
        self.dropratio = 0.1

        # optimize
        self.batch_size = 64
        self.batch_size_test = 10
        self.lr = 0.0001
        self.epochs = 5
        self.l2_weight = 0.00001
        self.runseed = 0

        localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.checkpoint_path = '../checkpoints/{}'.format(localtime)
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))

class load_SgCPI_dataset(InMemoryDataset):
    def __init__(self, root, tv_split_pth):
        self.root = root
        self.tv_split_pth = tv_split_pth

        self.protein_fasta_file = '../dataset/protein.fasta'
        self.protein_list = []

        with open(self.protein_fasta_file,'r') as f:
            text = f.readlines()
        for i in range(0,len(text),2):
            self.protein_list.append(text[i].strip()[1:])

        self.comp_list = pd.read_csv('../dataset/compound_smiles.csv')['cid'].tolist()

        super(load_SgCPI_dataset, self).__init__(root)
        self.data = torch.load(self.processed_dir + '/data.pt')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):

        prot_fea_dict = joblib.load('../dataset/protein_fea_dict.pt')
        comp_fea_dict = joblib.load('../dataset/compound_fea_dict.pt')

        prot_fea = []
        for prot_id in self.protein_list:
            prot_fea.append(torch.tensor(prot_fea_dict[prot_id][:-1]).float())  # grasr最后一维为序列长度
        prot_fea = torch.row_stack(prot_fea)

        comp_fea = []
        for comp_id in self.comp_list:
            comp_fea.append(torch.tensor(comp_fea_dict[str(int(comp_id))]).float())
        comp_fea = torch.row_stack(comp_fea)
        
        data = HeteroData()
        data['compound'].x = comp_fea
        data['protein'].x = prot_fea
        torch.save(data, self.processed_dir + '/data.pt')

    def load_edge(self, subset):
        metadata = self.data.metadata()
        for edge_type in metadata[1]:
            del self.data[edge_type]

        protein_id_dict = dict(zip(self.protein_list, list(range(len(self.protein_list)))))
        comp_id_dict = dict(zip(self.comp_list, list(range(len(self.comp_list)))))

        train_edge_index, train_label = self.load_subset_edge('train', comp_id_dict, protein_id_dict)

        train_pos_edge_index = train_edge_index[:, train_label == 1]
        train_neg_edge_index = train_edge_index[:, train_label == 0]

        if subset == 'train':
            self.data['compound', 'active', 'protein'].edge_index = train_pos_edge_index
            self.data['compound', 'inactive', 'protein'].edge_index = train_neg_edge_index
            self.data['compound', 'pred', 'protein'].edge_index = train_edge_index
            self.data['compound', 'pred', 'protein'].edge_label = train_label

        if subset == 'valid':
            valid_edge_index, valid_label = self.load_subset_edge('valid', comp_id_dict, protein_id_dict)
            self.data['compound', 'active', 'protein'].edge_index = train_pos_edge_index
            self.data['compound', 'inactive', 'protein'].edge_index = train_neg_edge_index
            self.data['compound', 'pred', 'protein'].edge_index = valid_edge_index
            self.data['compound', 'pred', 'protein'].edge_label = valid_label

        if subset == 'test':
            valid_edge_index, valid_label = self.load_subset_edge('valid', comp_id_dict, protein_id_dict)
            test_edge_index, test_label = self.load_subset_edge('test', comp_id_dict, protein_id_dict)

            valid_pos_edge_index = valid_edge_index[:, valid_label == 1]
            valid_neg_edge_index = valid_edge_index[:, valid_label == 0]
            train_valid_pos_edge_index = torch.cat([train_pos_edge_index, valid_pos_edge_index], dim=1)
            train_valid_neg_edge_index = torch.cat([train_neg_edge_index, valid_neg_edge_index], dim=1)

            self.data['compound', 'active', 'protein'].edge_index = train_valid_pos_edge_index
            self.data['compound', 'inactive', 'protein'].edge_index = train_valid_neg_edge_index

            self.data['compound', 'pred', 'protein'].edge_index = test_edge_index
            self.data['compound', 'pred', 'protein'].edge_label = test_label

        T.ToUndirected()(self.data)
        print('load ', subset)

    def load_subset_edge(self, subset, comp_id_dict, protein_id_dict):
        anno_df = pd.read_csv('{}/{}.txt'.format(self.tv_split_pth, subset), sep='\t')

        comp_list = anno_df['cid'].tolist()
        comp_list = [comp_id_dict[cid] for cid in comp_list]
        prot_list = anno_df['uniport'].tolist()
        prot_list = [protein_id_dict[uniport] for uniport in prot_list]

        CPI_edge_index = torch.tensor([comp_list, prot_list]).long()
        CPI_label = torch.tensor(anno_df['label'].tolist())

        return CPI_edge_index, CPI_label

class GNN_subgraph_noderep_AE(torch.nn.Module):
    def __init__(self, encoder,encoder_num_layers, hidden_channels, num_subgraph, prot_emb_dim, comp_emb_dim, dropratio):
        super().__init__()

        self.prot_emb_dim = prot_emb_dim

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

        self.prot_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels))

        self.comp_mlp = torch.nn.Sequential(
            torch.nn.Linear(comp_emb_dim, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropratio),
            torch.nn.Linear(hidden_channels, hidden_channels))

        self.encoder = encoder

        input_dim = hidden_channels * num_subgraph * (encoder_num_layers + 1)

        self.prot_rep = torch.nn.Sequential(torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(input_dim, hidden_channels))

        self.comp_rep = torch.nn.Sequential(torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(input_dim, hidden_channels))


    def forward(self, data_list, edge_label_index_list, emb=False):
        prot_emb_list = []
        comp_emb_list = []

        for i, (data, edge_label_index) in enumerate(zip(data_list, edge_label_index_list)):
            root_comp_index = edge_label_index[0]
            root_prot_index = edge_label_index[1]

            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            batch_dict = data.batch_dict

            prot_fea = x_dict['protein']

            esm_fea = prot_fea[:, :768]
            grasa_fea = prot_fea[:, 768:]
            esm_fea = self.esm_mlp(esm_fea)
            grasa_fea = self.grasr_mlp(grasa_fea)
            prot_fea = torch.cat([esm_fea, grasa_fea], dim=1)
            mlp_prot_emb = self.prot_mlp(prot_fea)

            mlp_comp_emb = self.comp_mlp(x_dict['compound'])

            x_dict['protein'] = mlp_prot_emb
            x_dict['compound'] = mlp_comp_emb

            prot_emb_list.append(x_dict['protein'][root_prot_index])
            comp_emb_list.append(x_dict['compound'][root_comp_index])

            if 'disease' in x_dict.keys():
                x_dict['disease'] = self.disease_mlp(x_dict['disease'])
            if 'se' in x_dict.keys():
                x_dict['se'] = self.se_mlp(x_dict['se'])

            x_dict_list = self.encoder(x_dict, edge_index_dict)

            for i in range(len(x_dict_list)):
                prot_emb_list.append(x_dict_list[i]['protein'][root_prot_index])
                comp_emb_list.append(x_dict_list[i]['compound'][root_comp_index])

        prot_emb = torch.cat(prot_emb_list, dim=-1)
        comp_emb = torch.cat(comp_emb_list, dim=-1)

        prot_emb = self.prot_rep(prot_emb)
        comp_emb = self.comp_rep(comp_emb)

        prot_emb_norm = F.normalize(prot_emb)
        comp_emb_norm = F.normalize(comp_emb)

        adj_pred = (torch.sum(prot_emb_norm * comp_emb_norm, dim=1) + 1) * 0.5

        if emb:
            return adj_pred, prot_emb, comp_emb
        else:
            return adj_pred

def valid(opt, model, device, valid_data, sample_graph=True, is_train=False):
    valid_edge_label_index = valid_data['compound', 'pred', 'protein'].edge_index
    valid_edge_label = valid_data['compound', 'pred', 'protein'].edge_label

    if sample_graph:
        subgraph_neighbors = opt.subgraph_neighbors
    else:
        subgraph_neighbors = opt.compgraph_neighbors

    print(subgraph_neighbors)
    dataloader = SubgraphLinkNeighborLoader(valid_data, num_neighbors=opt.num_neighbors, subgraph_neighbors=subgraph_neighbors,
                                                batch_size=1,directed=False, shuffle=False,
                                              edge_label_index=(('compound', 'pred', 'protein'), valid_edge_label_index),
                                              edge_label=valid_edge_label)
    model.to(device)
    model.eval()
    score_list = []
    pred_list = []
    target_list = []
    prot_emb_list = []
    comp_emb_list = []

    batch_list = [[] for i in range(opt.num_subgraph)]

    with torch.no_grad():
        for ii, subgraphs in enumerate(tqdm(dataloader)):
            for gi, subgraph in enumerate(subgraphs):
                batch_list[gi].append(subgraph)


            if (ii % opt.batch_size_test == 0 and ii != 0) or ii == (valid_edge_label_index.shape[1] - 1):
                processed_batch_list = []
                processed_edge_label_index = []
                processed_edge_label = []

                for gi, batch in enumerate(batch_list):
                    batch = Batch.from_data_list(batch).to(device)

                    if is_train:
                        edge_label_index = batch['compound', 'pred', 'protein'].edge_label_index  # 采样后batchsize个边
                        edge_label = batch['compound', 'pred', 'protein'].edge_label
                        edge1 = torch.cat([edge_label_index, edge_label.unsqueeze(0)], dim=0)

                        edge_index = torch.cat([batch['compound', 'active', 'protein'].edge_index, batch['compound', 'inactive', 'protein'].edge_index], dim=1)
                        edge_attr = torch.zeros((1, edge_index.shape[1])).to(edge_index.device)
                        edge_attr[:, : batch['compound', 'active', 'protein'].edge_index.shape[1]] = 1

                        edge2 = torch.cat([edge_index, edge_attr], dim=0)
                        combined = torch.cat((edge1, edge2), dim=1)

                        uniques, counts = combined.unique(return_counts=True, dim=1)
                        difference = uniques[:, counts == 1]
                        diff_edge_index = difference[:2, :].long()
                        diff_edge_attr = difference[2, :]

                        pos_edge_index = diff_edge_index[:, diff_edge_attr == 1]
                        neg_edge_index = diff_edge_index[:, diff_edge_attr == 0]

                        batch['compound', 'active', 'protein'].edge_index = pos_edge_index
                        batch['compound', 'inactive', 'protein'].edge_index = neg_edge_index
                        batch['protein', 'rev_active', 'compound'].edge_index = torch.row_stack(
                            [pos_edge_index[1], pos_edge_index[0]])
                        batch['protein', 'rev_inactive', 'compound'].edge_index = torch.row_stack(
                            [neg_edge_index[1], neg_edge_index[0]])

                        del batch['compound', 'pred', 'protein']
                        del batch['protein', 'rev_pred', 'compound']

                    else:
                        edge_label_index = batch['compound', 'pred', 'protein'].edge_label_index
                        edge_label = batch['compound', 'pred', 'protein'].edge_label

                        del batch['compound', 'pred', 'protein']
                        del batch['protein', 'rev_pred', 'compound']

                    processed_batch_list.append(batch)
                    processed_edge_label_index.append(edge_label_index)
                    processed_edge_label.append(edge_label)

                edge_label = processed_edge_label[0]
                score, prot_emb, comp_emb = model(processed_batch_list, processed_edge_label_index, emb=True)

                pred = score
                score = score.detach().cpu()
                prot_emb = prot_emb.detach().cpu()
                comp_emb = comp_emb.detach().cpu()
                score_list.append(score)
                prot_emb_list.append(prot_emb)
                comp_emb_list.append(comp_emb)
                target_list.extend(edge_label.detach().cpu().tolist())
                pred_list.extend(pred.detach().cpu().tolist())

                batch_list = [[] for i in range(opt.num_subgraph)]

    score_list = torch.cat(score_list, dim=0).numpy()
    prot_emb_list = torch.cat(prot_emb_list, dim=0).numpy()
    comp_emb_list = torch.cat(comp_emb_list, dim=0).numpy()
    target_list = np.array(target_list)
    pred_list = np.array(pred_list)

    if is_train:
        return score_list, target_list, prot_emb_list, comp_emb_list
    else:
        return pred_list, target_list

def train(opt, setting, fold):
    
    opt.tv_split_pth = '../dataset/{}/fold{}'.format(setting, fold)
    print('load ', opt.tv_split_pth)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(opt.runseed)
    np.random.seed(opt.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.runseed)

    loaddata = load_SgCPI_dataset(opt.root, opt.tv_split_pth)

    loaddata.load_edge('train')
    data = loaddata.data
    metadata = data.metadata()
    metadata[1].remove(('compound', 'pred', 'protein'))
    metadata[1].remove(('protein','rev_pred', 'compound'))

    prot_fea_dim = data['protein'].x.shape[1]
    comp_fea_dim = data['compound'].x.shape[1]

    train_edge_label_index = data['compound', 'pred', 'protein'].edge_index
    train_edge_label = data['compound', 'pred', 'protein'].edge_label

    encoder = HGNN(metadata=metadata, hidden_channels=opt.hidden_size, num_layers=opt.n_layers)
    model = GNN_subgraph_noderep_AE(encoder=encoder, encoder_num_layers=opt.n_layers, hidden_channels=opt.hidden_size, num_subgraph=opt.num_subgraph,
                         prot_emb_dim=prot_fea_dim, comp_emb_dim=comp_fea_dim,dropratio=opt.dropratio)

    criterion = torch.nn.BCELoss(reduction="mean")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2_weight)

    best_valid_auc = -1
    best_db_valid_auc = -1
    es = 0

    for epoch in range(0, opt.epochs + 1):
        loaddata.load_edge('train')
        data = loaddata.data
        train_loader = SubgraphLinkNeighborLoader(data, num_neighbors=opt.num_neighbors,
                                                  subgraph_neighbors=opt.subgraph_neighbors,
                                                  batch_size=1, replace=True,
                                                  directed=False, shuffle=True,
                                                  edge_label_index=(
                                                  ('compound', 'pred', 'protein'), train_edge_label_index),
                                                  edge_label=train_edge_label)

        model.train()

        train_pred_all = []
        train_target_all = []

        loss_meter = meter.AverageValueMeter()

        batch_list = [[] for i in range(opt.num_subgraph)]
        for ii, subgraphs in enumerate(tqdm(train_loader)):
            for gi, subgraph in enumerate(subgraphs):
                batch_list[gi].append(subgraph)

            if (ii % opt.batch_size == 0 and ii != 0) or ii == train_edge_label_index.shape[1] - 1:
                optimizer.zero_grad()

                processed_batch_list = []
                processed_edge_label_index = []
                processed_edge_label = []
                for gi, batch in enumerate(batch_list):
                    batch = Batch.from_data_list(batch).to(device)

                    edge_label_index = batch['compound', 'pred', 'protein'].edge_label_index
                    edge_label = batch['compound', 'pred', 'protein'].edge_label
                    edge1 = torch.cat([edge_label_index, edge_label.unsqueeze(0)], dim=0)

                    edge_index = torch.cat([batch['compound', 'active', 'protein'].edge_index, batch['compound', 'inactive', 'protein'].edge_index], dim=1)
                    edge_attr = torch.zeros((1, edge_index.shape[1])).to(edge_index.device)
                    edge_attr[:, : batch['compound', 'active', 'protein'].edge_index.shape[1]] = 1

                    edge2 = torch.cat([edge_index, edge_attr], dim=0)
                    combined = torch.cat((edge1, edge2), dim=1)

                    uniques, counts = combined.unique(return_counts=True, dim=1)
                    difference = uniques[:, counts == 1]
                    diff_edge_index = difference[:2, :].long()
                    diff_edge_attr = difference[2, :]

                    pos_edge_index = diff_edge_index[:,diff_edge_attr==1]
                    neg_edge_index = diff_edge_index[:,diff_edge_attr==0]

                    batch['compound', 'active', 'protein'].edge_index = pos_edge_index
                    batch['compound', 'inactive', 'protein'].edge_index = neg_edge_index
                    batch['protein', 'rev_active', 'compound'].edge_index = torch.row_stack([pos_edge_index[1], pos_edge_index[0]])
                    batch['protein', 'rev_inactive', 'compound'].edge_index = torch.row_stack([neg_edge_index[1], neg_edge_index[0]])

                    del batch['compound', 'pred', 'protein']
                    del batch['protein', 'rev_pred', 'compound']

                    processed_batch_list.append(batch)
                    processed_edge_label_index.append(edge_label_index)
                    processed_edge_label.append(edge_label)

                edge_label = processed_edge_label[0].float()
                train_pred = model(processed_batch_list, processed_edge_label_index)
                loss = criterion(train_pred, edge_label)
                loss.backward()
                optimizer.step()
                loss_meter.add(loss.item())
                train_pred_all.append(train_pred)
                train_target_all.append(edge_label)

                batch_list = [[] for i in range(opt.num_subgraph)]

        print('|| Epoch {} | train loss={:.5f} ||'.format(epoch, loss_meter.mean))
        loss_meter.reset()
        torch.cuda.empty_cache()

        loaddata.load_edge('valid')
        data = loaddata.data
        valid_pred, valid_label = valid(opt, model, device, data)
        val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_ap, re0_5, re1, re2, re5, pred_class = eval_metrics(valid_pred, valid_label)
        print('valid result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
              .format(val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_ap))

        loaddata.load_edge('test')
        data = loaddata.data
        test_pred, test_label = valid(opt, model, device, data)
        test_th, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc, test_ap, re0_5, re1, re2, re5, pred_class = th_eval_metrics(val_th, test_pred, test_label)
        print('test result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
              .format(test_th, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc, test_ap))

        if val_auc > best_valid_auc:
            es = 0
            save_path = '{}/model_epoch{}.pth'.format(opt.checkpoint_path, fold, epoch)
            print('save net: ', save_path)
            torch.save(model.state_dict(), save_path)
            
            best_valid_auc = val_auc
            best_valid_results = [val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_ap]
            best_test_results = [test_th,test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc, test_ap]
            print('## EPOCH {} || higher AUC of valid'.format(epoch))
        else:
            es += 1

        if es == 10:
            break

    print('===================Result==================='.format(fold))
    print('valid result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
          .format(best_valid_results[0], best_valid_results[1], best_valid_results[2], best_valid_results[3], best_valid_results[4], best_valid_results[5], best_valid_results[6], best_valid_results[7]))
    print('test result: th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} AP={:.3f}'
          .format(best_test_results[0], best_test_results[1], best_test_results[2], best_test_results[3], best_test_results[4], best_test_results[5], best_test_results[6], best_test_results[7]))

    return


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--setting", dest="setting", help="choose from transductive/semi-inductive/inductive.")
    parser.add_argument("--fold", dest="fold", help="choose from 0 to 4.")
    return parser.parse_args()

if __name__ == "__main__":
    opt = Config()
    opt.print_config()

    args = parse_args()
    setting = args.setting
    fold = args.fold

    train(opt, setting, fold)
