import torch
from torch import nn
from torch.nn.parameter import Parameter
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.mm(adj.squeeze(-1), seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class classifier(nn.Module):
    def __init__(self, in_features, nb_classes):
        super(classifier, self).__init__()
        self.fc = nn.Linear(in_features, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features):
        ret = self.fc(features)
        return ret


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# Latent Relation Exploration
class LatentRelationExploration(nn.Module):
    def __init__(self,
                 in_features: int,
                 n_hidden_gcn: int,
                 n_classes: int,
                 n_heads: int,
                 n_layers: int = 2,
                 n_clusters: int = 2,
                 is_concat: bool = True,
                 share_weights: bool = False):

        super().__init__()

        self.heads = n_heads
        self.layers = n_layers
        self.is_concat = is_concat
        self.share_weights = share_weights

        # gcn
        self.gcn = GCN(in_features, n_hidden_gcn)

        # classifier
        self.classifier = classifier(n_hidden_gcn, n_classes)

        # A_i
        self.linear_query = nn.Linear(in_features, in_features, bias=False)
        self.linear_key = nn.Linear(in_features, in_features, bias=False)
        self.softmax = nn.Softmax(dim=1)

        def derivative_adjacency_matrix(features, a_0):
            query = self.linear_query(features)
            key = self.linear_key(features)
            dot_product = torch.matmul(query, key.transpose(0, 1)) / torch.sqrt(
                torch.tensor(features.shape[1], dtype=features.dtype))
            return self.softmax(dot_product) * a_0

        self.derivative_adjacency_matrix = derivative_adjacency_matrix

        # dense connection
        self.dense_connection = nn.ModuleList([
            nn.ModuleList([
                GCN(in_features + layer * (in_features // self.layers), in_features // self.layers, bias=True)
                for layer in range(n_layers)
            ])
            for _ in range(n_heads)
        ])
        self.linear_h = nn.Linear(in_features * 2 * n_heads, in_features, bias=True)

        # clustering layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_hidden_gcn))

        # q
        self.alpha = 0.0001

    def forward(self, features: torch.Tensor, adj: torch.Tensor):
        # latent relation exploration
        A = [self.derivative_adjacency_matrix(features, adj) for _ in range(self.heads)]

        # densely connected layers
        h = []
        for i, head in enumerate(self.dense_connection):
            v_m_l = [features]
            for layer, gcn in enumerate(head):
                h_m_l = gcn(v_m_l[0], A[i])
                v_m_l.append(h_m_l)
                v_m_l = [torch.cat(v_m_l, dim=1)]
            h.append(v_m_l[0])
        h = self.linear_h(torch.cat(h, dim=1))

        # feature representation
        feats_gcn = self.gcn(features, adj)
        latent_gcn = self.gcn(h, adj)

        # Q distribution of clustering layer
        q = 1.0 / (1.0 + torch.pow(
            torch.cdist(feats_gcn, self.cluster_layer), 2
        ) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)

        return feats_gcn, latent_gcn, q
