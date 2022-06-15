import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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
        return self.act(output)


''' Simplifying Quaternion graph networks! '''
class SQGNLayer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh, step_k=1):
        super(SQGNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.step_k = step_k
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        #self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        new_input = torch.spmm(adj, input)
        if self.step_k > 1:
            for _ in range(self.step_k-1):
                new_input = torch.spmm(adj, new_input)
        output = torch.mm(new_input, hamilton)  # Hamilton product, quaternion multiplication!
        #output = self.bn(output)
        return output

''' Quaternion Graph Isomorphism Networks! QGNN layer! '''
class QGINLayer(Module):
    def __init__(self, in_features, out_features, hid_size, act=torch.tanh):
        super(QGINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_size = hid_size
        self.act = act
        #
        self.weight1 = Parameter(torch.FloatTensor(self.in_features // 4, self.hid_size))
        self.weight2 = Parameter(torch.FloatTensor(self.hid_size // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(self.hid_size)
        #self.bn2 = torch.nn.BatchNorm1d(self.out_features)

    def reset_parameters(self):
        stdv1 = math.sqrt(6.0 / (self.weight1.size(0) + self.weight1.size(1)))
        self.weight1.data.uniform_(-stdv1, stdv1)

        stdv2 = math.sqrt(6.0 / (self.weight2.size(0) + self.weight2.size(1)))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, input, adj):
        hamilton1 = make_quaternion_mul(self.weight1)
        hamilton2 = make_quaternion_mul(self.weight2)
        new_input = torch.spmm(adj, input)
        output1 = torch.mm(new_input, hamilton1)  # Hamilton product, quaternion multiplication!
        output1 = self.bn(output1)
        output1 = self.act(output1)
        output2 = torch.mm(output1, hamilton2)
        #output2 = self.bn(output2)
        return output2


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu,  bias=False):
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

