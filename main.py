from load_data import Data
import numpy as np
import torch
from collections import defaultdict
import argparse
from utils_WGE import construct_entity_focus_matrix, construct_relation_focus_matrix, get_deg, get_er_vocab, get_batch
from model import *
from tqdm import tqdm

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True
np.random.seed(1337)


class WGE:
    """ Two-view Graph Neural Networks for Knowledge Graph Completion """
    def __init__(self):
        self.args = args


    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def evaluate(self, model, data, lst_indexes1, lst_indexes2, er_vocab, test=False, save=False, ep=0):
        model.eval()
        rel_dict = defaultdict(list)
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            print("Number of data points: %d" % len(test_data_idxs))
            # import pdb; pdb.set_trace()


            for i in range(0, len(test_data_idxs), 1000):
                data_batch, _ = get_batch(er_vocab, test_data_idxs, i, 1000, d)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                forward = model.forward_normal
                
                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False)

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)
                
                pred = 0
                for _layer in range(num_preds):
                    pred += preds[_layer] * weights[_layer]
                    
                pred = pred.detach()

                for j in range(data_batch.shape[0]):
                    this_e1 = data_batch[j][0]
                    this_r = data_batch[j][1]
                    filt = er_vocab[(this_e1, this_r)]
                    target_value = pred[j, e2_idx[j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()

                for j in range(data_batch.shape[0]):
                    this_r = data_batch[j][1]
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    rel_dict[self.relation_idx2id[this_r]].append(rank)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)


        hit1 = np.mean(hits[0])*100
        hit3 = np.mean(hits[2])*100
        hit5 = np.mean(hits[4])*100
        hit10 = np.mean(hits[9])*100
        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        print('Hits @1: {0}'.format(hit1))
        print('Hits @3: {0}'.format(hit3))
        print('Hits @5: {0}'.format(hit5))
        print('Hits @10: {0}'.format(hit10))
        print('MR: {0}'.format(MR))
        print('Mean reciprocal rank: {0}'.format(MRR))

        return hit1, hit3, hit10, MR, MRR


    def prepare_data(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))} # entity ids to index
        self.entity_idx2id = {v:k for k,v in self.entity_idxs.items()}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))} # relation ids to index
        self.relation_idx2id = {v:k for k,v in self.relation_idxs.items()}
        # import pdb; pdb.set_trace()

        adj_r = construct_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)
        adj = construct_entity_focus_matrix(d.train_data, self.entity_idxs)

        train_data_idxs = self.get_data_idxs(d.train_data)
        deg = get_deg(train_data_idxs, len(self.entity_idxs)).to(device)

        print("Number of training data points: %d" % len(train_data_idxs))

        model = WGE_model(args, len(self.entity_idxs), len(self.relation_idxs), adj, adj_r, deg=deg).to(device)
        print("Using Adam optimizer")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        lst_indexes1 = torch.LongTensor([i for i in range(len(d.entities))]).to(device)
        lst_indexes2 = torch.LongTensor([i for i in range(len(d.entities) + len(d.relations))]).to(device)
        er_vocab = get_er_vocab(train_data_idxs)
        er_vocab_all = get_er_vocab(self.get_data_idxs(d.data))
        er_vocab_pairs = list(er_vocab.keys())

        return model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs


    def train_and_eval(self, model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs):
        
        max_valid_mrr, best_hit10_ever, best_mrr_ever, best_epoch = 0.0, 0.0, 0.0, 0
        best_valid_hit10_ever, best_valid_mrr_ever = 0.0, 0.0
        forward = model.forward_normal

        print("Starting training...")
        for it in tqdm(range(1, args.num_iterations + 1)):
            model.train()            
            losses, losses0, losses1, losses2 = [], [], [], []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), args.batch_size):
                data_batch, targets = get_batch(er_vocab, er_vocab_pairs, j, args.batch_size, d)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=True)
                targets = ((1.0 - args.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss_list = [0, 0, 0]
                loss = 0

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)
                
                for _layer in range(num_preds):
                    this_loss = model.loss(preds[_layer], targets)
                    loss_list[_layer] = this_loss.item()
                    loss += this_loss * weights[_layer]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
                losses0.append(loss_list[0])
                losses1.append(loss_list[1])
                losses2.append(loss_list[2])
            
            print("Epoch: {} --> Loss: {:.4f}, Loss0: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}".format(it, np.sum(losses), np.sum(losses0), np.sum(losses1), np.sum(losses2)))

            if (it > args.eval_after and it % args.eval_step == 0) or (it == 1):
                print("Valid:")
                hit1, hit3, hit10, MR, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes1, lst_indexes2, er_vocab_all, test=False, save=True, ep=it)
                if max_valid_mrr < tmp_mrr:
                    max_valid_mrr = tmp_mrr
                    best_epoch = it
                if best_valid_hit10_ever < hit10:
                    best_valid_hit10_ever = hit10 
                
                print("Test:")
                t_hit1, t_hit3, t_hit10, t_MR, t_MRR = self.evaluate(model, d.test_data, lst_indexes1, lst_indexes2, er_vocab_all, test=True, save=True, ep=it)
                if best_hit10_ever < t_hit10:
                    best_hit10_ever = t_hit10
                if best_mrr_ever < t_MRR:
                    best_mrr_ever = t_MRR
                        
                print("Best valid epoch", best_epoch, " --> Final test results: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(t_hit1, t_hit3, t_hit10, t_MR, t_MRR))
                #print("x" * 30, "Best test h10 ever: {:.4f}, best test mrr: {:.4f}".format(best_hit10_ever, best_mrr_ever), "x" * 30) 
                #print("x" * 30, "Best valid h10 ever: {:.4f}, best valid mrr: {:.4f}".format(best_valid_hit10_ever, max_valid_mrr), "x" * 30) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="codex-s", nargs="?", help="codex-s, codex-m, and codex-l")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?", help="Batch size.")
    parser.add_argument("--emb_dim", type=int, default=256, nargs="?", help="embedding size")
    parser.add_argument("--encoder", type=str, default="qgnn", nargs="?", help="encoder")
    parser.add_argument("--decoder", type=str, default="quate", nargs="?", help="decoder")
    parser.add_argument("--eval_step", type=int, default=10, nargs="?", help="how often doing eval")
    parser.add_argument("--eval_after", type=int, default=500, nargs="?", help="only eval after this interation")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="smoothing constant")

    parser.add_argument("--num_iterations", type=int, default=3000, nargs="?", help="Number of iterations.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?", help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--first_layer_weight", type=float, default=0.9, help="weight of the first GCN layer") # weight of the first GCN layer!
    parser.add_argument("--beta", type=float, default=0.2, help="triple keeping percentage")
    parser.add_argument("--combine_type", type=str, default='corr', help="cat, average, corr, linear_corr") # consider this

    args = parser.parse_args()
    print(args)

    # load dataset
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    d = Data(data_dir=data_dir) 

    gnnkge = WGE()

    model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs = gnnkge.prepare_data()
    gnnkge.train_and_eval(model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs)

