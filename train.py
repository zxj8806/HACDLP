# set CUDA_LAUNCH_BLOCKING=1
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import torch.nn as nn

import math
from input_data import load_data
from preprocessing import *
import args

from tqdm.auto import tqdm

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import sys
import os

# Disable buffering of stdout
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)

adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                    torch.FloatTensor(features[1]),
                                    torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight


def linear_beta_schedule(timestesps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timestesps)


REDUCED_DIM = 1
timesteps = args.n
betas = linear_beta_schedule(timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Compute previous step alphas cumulative product
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# Calculations for diffusion q(x_t | x_0) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)  # noise3

cumulative_sum_reversed = torch.flip(sqrt_one_minus_alphas_cumprod, dims=[0]).cumsum(dim=0)
cumulative_sqrt_one_minus_alphas_cumprod = torch.flip(cumulative_sum_reversed, dims=[0])  # noise4

# Calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# print("posterior_variance", posterior_variance)
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def get_value_at_index(tensor, index):
    index = index - 1
    if index < 0 or index >= tensor.numel():
        raise IndexError("Index out of bounds")

    return tensor[index].item()


def D_losses(denoise_model, adj_label, features, t, noise=None, loss_type="huber"):
    predicted_adj_t, q_z_list, p_z_list = denoise_model(features.to(device), t, timesteps)

    if predicted_adj_t.device != device:
        predicted_adj_t = predicted_adj_t.to(device)

    # Calculate KL Divergence and average it over the number of timesteps
    kl_divergence = 0
    for q_z, p_z in zip(q_z_list, p_z_list):
        kl_divergence += torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    kl_divergence /= args.n  # Divide the cumulative KL divergence by the number of timesteps

    # Calculate total loss
    if loss_type == 'L1':
        total_loss = F.l1_loss(adj_label.to(device), predicted_adj_t)
    elif loss_type == 'L2':
        total_loss = F.mse_loss(adj_label.to(device), predicted_adj_t)
    elif loss_type == 'huber':
        predicted_adj_t = torch.clamp(predicted_adj_t, 0, 1)
        total_loss = F.binary_cross_entropy(predicted_adj_t.view(-1), adj_label.to(device).to_dense().view(-1),
                                            weight=weight_tensor.to(device))
    else:
        raise NotImplementedError("Loss type not implemented")

    total_loss += args.sample * kl_divergence  # Include the averaged KL divergence in the total loss

    return total_loss, predicted_adj_t


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias


def positional_encoding(t, T, d_model, device):
    position = torch.arange(0, d_model, dtype=torch.float, device=device).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(1, d_model, device=device)
    pe[:, 0::2] = torch.sin(position[:, 0::2] * div_term)
    pe[:, 1::2] = torch.cos(position[:, 1::2] * div_term)
    return pe


class GraphkGNNConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphkGNNConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = x + torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def compute_spherical_adjacency_matrix(a_i):
    a_i = a_i - a_i.mean(dim=0, keepdim=True)

    a_i = F.normalize(a_i, p=2, dim=1)

    dot_product_matrix = torch.mm(a_i, a_i.t())

    print("dot_product_matrix", dot_product_matrix)
    dot_product_matrix = torch.clamp(dot_product_matrix, -1.0 + 1e-9, 1.0 - 1e-9)

    angle_matrix = torch.acos(dot_product_matrix)

    adjacency_matrix = torch.sigmoid(angle_matrix)

    return adjacency_matrix


class GraphHACDLP(nn.Module):
    def __init__(self, adj):
        super(GraphHACDLP, self).__init__()
        self.n = args.n + 1
        self.gcns_a1 = nn.ModuleList(
            [GraphkGNNConvSparse(args.input_dim, args.hidden1_dim, adj) for _ in range(self.n)])
        self.gcns_a2 = nn.ModuleList(
            [GraphkGNNConvSparse(args.hidden1_dim, args.hidden2_dim, adj) for _ in range(self.n)])
        self.gcn_mean = GraphkGNNConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_concentration = GraphkGNNConvSparse(args.hidden1_dim, 1, adj, activation=lambda x: F.softplus(x) + 1)
        self.gcns_concentration = nn.ModuleList(
            [GraphkGNNConvSparse(args.hidden1_dim, 1, adj, activation=lambda x: F.softplus(x) + 1) for _ in
             range(self.n)])

    def encode(self, X, t):
        X = X.to(device)  # Ensure X is on the correct device before processing
        hidden = self.gcns_a1[t](X)
        mean = self.gcns_a2[t](hidden)
        mean = F.normalize(mean, p=2, dim=1)  # Normalize to ensure it lies on the unit sphere
        concentration = self.gcns_concentration[t](hidden)
        return mean.to(device), concentration.to(device)

    def reparameterize(self, mean, concentration):
        # Ensure tensors are on the right device - should be redundant if encode handles this
        mean = mean.to(device)
        concentration = concentration.to(device)
        q_z = VonMisesFisher(mean, concentration)  # Create distribution with parameters on the correct device
        p_z = HypersphericalUniform(mean.size(1) - 1, device=device)  # Explicitly set the device for the distribution
        return q_z, p_z

    def forward(self, X, time_step, timesteps):
        predicted_adj = torch.zeros(X.size(0), X.size(0), device=X.device)
        sqrt_one_minus_alphas_cumprod_t = get_value_at_index(cumulative_sqrt_one_minus_alphas_cumprod, time_step)

        q_z_list = []  # To store q_z for each time step
        p_z_list = []  # To store p_z for each time step

        for t in range(time_step, timesteps + 1):
            sqrt_alphas_cumprod_t = get_value_at_index(sqrt_one_minus_alphas_cumprod, t)
            mean, concentration = self.encode(X, t)
            q_z, p_z = self.reparameterize(mean, concentration)
            sampled_z = q_z.rsample()  # Sample from the Von Mises-Fisher distribution
            mean = F.normalize(mean, dim=1)

            # perturbed_mean = sampled_z
            perturbed_mean = mean + args.sample * sampled_z

            # perturbed_mean = mean
            perturbed_mean = F.normalize(perturbed_mean, dim=1)  # Re-normalize to ensure it lies on the unit sphere

            adj_t = torch.mm(perturbed_mean, perturbed_mean.t())
            predicted_adj += sqrt_alphas_cumprod_t * adj_t

            # Store distributions
            q_z_list.append(q_z)
            p_z_list.append(p_z)

        predicted_adj /= sqrt_one_minus_alphas_cumprod_t  # Normalize the predicted_adj matrix

        # Return the predicted adjacency matrix and the distributions
        return predicted_adj, q_z_list, p_z_list


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


@torch.no_grad()
def sample(model, features, adj_label, start_t, device):
    predicted_adj_t, q_z_list, p_z_list = model(features, start_t, timesteps)

    if predicted_adj_t.device != device:
        predicted_adj_t = predicted_adj_t.to(device)

    if predicted_adj_t.is_sparse:
        predicted_adj_t = predicted_adj_t.to_dense()
    predicted_adj_t = torch.clamp(predicted_adj_t, 0, 1)

    return predicted_adj_t


print("torch.cuda.is_available()", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

adj_norm = adj_norm.to(device)

model = GraphHACDLP(adj_norm)
model.to(device)
features = features.to(device)
adj_label = adj_label.to(device)
weight_tensor = weight_tensor.to(device)
cumulative_sqrt_one_minus_alphas_cumprod = cumulative_sqrt_one_minus_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
epochs = args.num_epoch

for epoch in range(epochs):
    optimizer.zero_grad()

    t = torch.randint(1, timesteps, (1,), device=device).long()

    loss, A_pred = D_losses(model, adj_label, features, t, loss_type="huber")

    loss.backward()
    optimizer.step()
    if t.item() < 10000:

        train_acc = get_acc(A_pred, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
        print("Epoch:", '%04d' % (epoch + 1), "time_step=", t.item(), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap))

    else:
        print(f"Epoch: {epoch + 1}, Time step: {t.item()}, Loss: {loss.item()}")

A_pred = sample(model, features, adj_label, 1, device)
test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

# torch.save(model, './model.pth')





