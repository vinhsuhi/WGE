import numpy as np
import torch
from torch.nn.init import xavier_normal_
import torch
from torch.nn import Parameter, Module
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *


def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion



def vec_vec_wise_multiplication(q, p):  # vector * vector
    normalized_p = normalization(p)  # bs x 4dim
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)



def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

''' The re-implementation of Quaternion Knowledge Graph Embeddings (https://arxiv.org/abs/1904.10281), following the 1-N scoring strategy '''
class QuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(QuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = vec_vec_wise_multiplication(h, r)
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred


def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2


def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
    quaternion = quaternion.reshape(-1, 4 * size)
    return quaternion


''' Quaternion graph neural networks! QGNN layer! https://arxiv.org/abs/2008.05089 '''
class Q4GNNLayer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(Q4GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)
        if self.act is not None:
            return self.act(output)
        else:
            return output

''' Quaternion Graph Isomophism Network!'''
class Q4GIN0Layer(Module):
    def __init__(self, in_features, hid_feature, out_features, act=torch.tanh):
        super(Q4GIN0Layer, self).__init__()
        self.in_features = in_features
        self.hid_feature = hid_feature
        self.out_features = out_features
        self.act = act
        #
        self.weight1 = Parameter(torch.FloatTensor(self.in_features // 4, self.hid_feature))
        self.weight2 = Parameter(torch.FloatTensor(self.hid_feature // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight1.size(0) + self.weight1.size(1)))
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton1 = make_quaternion_mul(self.weight1)
        hamilton2 = make_quaternion_mul(self.weight2)
        output1 = torch.mm(input, hamilton1)  # Hamilton product, quaternion multiplication!
        if self.act is not None:
            output1 = self.act(output1)
        output2 = torch.mm(output1, hamilton2)

        output = torch.spmm(adj, output2)
        output = self.bn(output)
        return output


class WGE_model(torch.nn.Module):
    def __init__(self, args, num_ents, num_rels, adj, adj_r, deg):
        super(WGE_model, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.emb_dim = args.emb_dim
        self.n_entities = num_ents
        self.n_relations = num_rels
        self.deg = deg

        self.thetas = nn.Parameter(torch.ones(3))

        self.all_embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
        # import pdb; pdb.set_trace()
        self.adj = adj 
        self.adj_r = adj_r

        self.lst_gcn1 = torch.nn.ModuleList()
        self.lst_gcn2 = torch.nn.ModuleList()
        for _layer in range(args.num_layers):
            if self.args.encoder == "qgnn":
                self.lst_gcn1.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
                self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
            else:
                print("This encoder has not been implemented... Existing")
                exit()

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()
        
        for _layer in range(args.num_layers):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.emb_dim // 2, self.emb_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.emb_dim // 4, self.emb_dim)))
        
        # if not self.args.best_model:
        self.reset_parameters()


        xavier_normal_(self.all_embeddings.weight.data)
        # import pdb; pdb.set_trace()

        self.activate = torch.nn.Tanh()

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def reset_parameters(self):
        for i in range(len(self.linear_ents)):
            weight = self.linear_ents[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents[i].data.uniform_(-stdv, stdv)

        for i in range(len(self.linear_ents_cor)):
            weight = self.linear_ents_cor[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents_cor[i].data.uniform_(-stdv, stdv)
        

    def forward_normal(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
        self.scorer = self.quate
        
        X = self.all_embeddings(lst_indexes1)
        R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])
        h1 = X[e1_idx]
        r1 = R[r_idx]
        hs = [h1]
        rs = [r1]

        scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]

        for _layer in range(self.args.num_layers):
            XR = torch.cat((X, R), dim=0) # last 
            XRrf = self.lst_gcn2[_layer](XR, self.adj_r) # newX, newR from relational graph
            Xef = self.lst_gcn1[_layer](X, self.adj) # newX2 from original graph 
            Xrf = XRrf[lst_indexes1]
            if self.args.combine_type == "cat":
                size = Xrf.size(1) // 4
                Xef1, Xef2, Xef3, Xef4 = torch.split(Xef, size, dim=1)
                Xrf1, Xrf2, Xrf3, Xrf4 = torch.split(Xrf, size, dim=1)
                X = torch.cat([Xef1, Xrf1, Xef2, Xrf2, Xef3, Xrf3, Xef4, Xrf4], dim=1)
                hamilton = make_quaternion_mul(self.linear_ents[_layer])
                X = torch.mm(X, hamilton)
            elif self.args.combine_type == "sum":
                X = Xef + Xrf
            elif self.args.combine_type == "corr":
                X = Xef * Xrf
            elif self.args.combine_type == "linear_corr":
                hamilton = make_quaternion_mul(self.linear_ents_cor[_layer])
                X = Xef * torch.mm(Xrf, hamilton)

            R = XRrf[lst_indexes2[len(lst_indexes1):]] # newR
            hs.append(X[e1_idx]) # finalX
            rs.append(R[r_idx]) # finalR
            scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

        return scores


    def quate(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr) 
            hr = self.dropout1(hr) 
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt


    def distmult(self, h, r, X, layer_index=1):
        hr = h * r 
        if layer_index == 1:
            hr = self.bn1(hr) 
            hr = self.dropout1(hr) 
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt 


    def transe(self, h, r, X, layer_index=1):
        hr = h + r 
        hrt = 20 - torch.norm(hr.unsqueeze(1) - X, p=1, dim=2)
        return hrt


    def normalize_embedding(self):
        embed = self.all_embeddings.weight.detach().cpu().numpy()[:self.n_entities]
        rel_emb = self.all_embeddings.weight.detach().cpu().numpy()[self.n_entities:]
        embed = embed / np.sqrt(np.sum(np.square(embed), axis=1, keepdims=True))
        
        self.all_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((embed, rel_emb), axis=0)))


    def get_hidden_feature(self):
        return self.feat_list


    def regularization(self, dis_loss, margin=1.5):
        return max(0, margin - dis_loss)


    def get_factor(self):
        factor_list = []
        factor_list.append(self.distangle.get_factor())
        return factor_list


    def compute_disentangle_loss(self):
        return self.distangle.compute_disentangle_loss()


    @staticmethod
    def merge_loss(dis_loss):
        return dis_loss



""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.tanh,  bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)

    # def forward_vew2_relation(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
    #     self.scorer = self.quate
    #     X = self.all_embeddings(lst_indexes1)
    #     R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])

    #     h1 = X[e1_idx]
    #     r1 = R[r_idx]
    #     hs = [h1]
    #     rs = [r1]

    #     if self.args.use_multiple_layers:
    #         scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]
    #     else:
    #         scores = []

    #     for _layer in range(self.args.num_layers):
    #         R = self.lst_gcn2[_layer](R, self.adj_r) # newX, newR from relational graph
    #         X = self.lst_gcn1[_layer](X, self.adj) # newX2 from original graph 
    #         hs.append(X[e1_idx]) # finalX
    #         rs.append(R[r_idx]) # finalR
    #         scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))
        
    #     return scores

    # def forward_entity_graph(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
    #     self.scorer = self.quate
    #     X = self.all_embeddings(lst_indexes1)
    #     R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])

    #     h1 = X[e1_idx]
    #     r1 = R[r_idx]
    #     hs = [h1]
    #     rs = [r1]

    #     if self.args.use_multiple_layers:
    #         scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]
    #     else:
    #         scores = []
    #     for _layer in range(self.args.num_layers):
    #         X = self.lst_gcn1[_layer](X, self.adj)
    #         r = r1 
    #         hs.append(X[e1_idx])
    #         rs.append(r)
    #         scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

    #     return scores