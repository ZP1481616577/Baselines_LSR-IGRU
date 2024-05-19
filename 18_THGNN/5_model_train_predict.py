import os
import sys
import math
import torch
import pickle
import warnings
import torch.multiprocessing
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import StepLR


warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

stock_name = 'sp500'
inputSize = 5

class Args:
    def __init__(self, gpu=0, subtask="regression"):
        # device
        self.gpu = str(gpu)
        self.device = 'cuda:0'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 30
        self.epochs_eval = 1
        # learning rate settings
        self.lr = 0.00003
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 512
        self.loss_fcn = mse_loss
        # save model settings
        self.save_path = os.path.join(os.path.abspath('.'), f"{stock_name}/model_saved/")
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)
        self.epochs_save_by = 30
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"


class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        masked_weight = torch.mul(weight, adj_mat).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GraphAttnSemIndividual(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, requires_weight=False):
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        if requires_weight:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=inputSize, out_features=8, num_heads=8, hidden_dim=64, num_layers=1):
        super(StockHeteGAT, self).__init__()
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=False):
        _, support = self.encoding(inputs)
        support = support.squeeze()
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        all_embedding = self.pn(all_embedding)
        if requires_weight:
            return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
        else:
            return self.predictor(all_embedding)


class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None, data_start=None, data_middle=None, data_end=None, mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
        if mode == "train":
            self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
        elif mode == "val":
            self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            data_all.append(pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb")))
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]


def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels, mask, loss_func=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos, adj_neg)

    loss = loss_func(logits,labels)
    return loss, logits


def extract_data(data_dict, device):
    pos_adj = data_dict['pos_adj'].to(device).squeeze()
    neg_adj = data_dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx == 0:
                loss_return += loss.data
    return loss_return/len(dataset_train)


def eval_epoch(args, model, dataset_eval, loss_fcn):
    loss = 0.
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
        loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        break
    return loss, logits


def fun_train_predict(data_start, data_middle, data_end, pre_data):
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dataset = AllGraphDataSampler(base_dir=f"{stock_name}/data_train_predict/", data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    val_dataset = AllGraphDataSampler(base_dir=f"{stock_name}/data_train_predict/", mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=mse_loss)
        if epoch % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=mse_loss)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss,
                                                                              eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))
        if (epoch + 1) % args.epochs_save_by == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))

    # predict
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
    model.load_state_dict(checkpoint['model'])
    data_kdcode = os.listdir(f'{stock_name}/kdcode')
    data_kdcode = sorted(data_kdcode)
    data_kdcode_last = data_kdcode[data_middle:data_end]
    df_score=pd.DataFrame()
    for i in tqdm(range(len(val_dataset))):
        df = pd.read_csv(f'{stock_name}/kdcode/' + data_kdcode_last[i], dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.train()
        logits = model(features, pos_adj, neg_adj)
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df_score=pd.concat([df_score,df])

        #df.to_csv('prediction/' + data_kdcode_last[i], encoding='utf-8-sig', index=False)
    df_score.to_csv(f'{stock_name}/prediction/pred.csv')
    print(df_score)

if __name__ == '__main__':
    # nasdaq&sp 0 1489-249=1240 249  hs&zz 0 1438-242=1196 242
    data_start = 0
    data_middle = 1240
    data_end = data_middle + 249
    pre_data = '2022-12-31'
    fun_train_predict(data_start, data_middle, data_end, pre_data)