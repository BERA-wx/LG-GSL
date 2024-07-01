import scipy as sp
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, diags
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment as linear_assignment


seed = 42


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def symmetric_normalized_Laplacian(edge, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        edge = edge + sp.eye(edge.shape[0])
    edge = coo_matrix(edge)
    rowsum = np.array(edge.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    return edge.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    # Use the following code to calculate accuracy
    accuracy_sum = sum([w[i, j] for i, j in zip(ind[0], ind[1])])
    return accuracy_sum * 1.0 / y_pred.size


def kolmogorov_smirnov(labels, pred):
    fpr, tpr, thresholds = roc_curve(labels, pred)
    ks_value = max(tpr - fpr)
    return ks_value


'''load data'''


# Lending Club
class LendingClubDatasets(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # specified index sample
        features = self.data.iloc[index, :-2].values.astype(np.float32)
        labels = self.data.iloc[index, -2].astype(np.float32)
        flags = self.data.iloc[index, -1].astype(np.float32)
        return features, labels, flags


def load_LendingClub(dataset_name, p=0.0, BATCH_SIZE=1000):
    """
    get features_dn and labels_dn：
    - features_dn: float 32
    - labels_dn: float 32
    """
    # data
    dataset_paths = {
        'Lending1': 'data/datasets/Lending1.csv',
        'Lending2': 'data/datasets/Lending2.csv',
        'Lending3': 'data/datasets/Lending3.csv',
    }

    if dataset_name in dataset_paths:
        path = dataset_paths[dataset_name]
    else:
        # handle invalid dataset_name case
        path = None

    feats_labels_flags = pd.read_csv(path)
    dataset = LendingClubDatasets(feats_labels_flags)

    # batch
    batch_size = BATCH_SIZE

    def calculate_adjacency_matrix(input_batch):
        # features & labels
        feats = csr_matrix(np.stack([item[0] for item in input_batch], axis=0), dtype=np.float32)
        lbs = encode_onehot(np.stack([item[1] for item in input_batch], axis=0))
        flags = encode_onehot(np.stack([item[2] for item in input_batch], axis=0))

        # calculate the adjacency matrix
        similarity_matrix = cosine_similarity(feats)
        threshold = p
        adj = csr_matrix(similarity_matrix > threshold, dtype=int)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # normalization
        adj = symmetric_normalized_Laplacian(adj)  # adj = D^{-0.5}SD^{-0.5}, S=A+I

        # format transformation
        feats = torch.FloatTensor(np.array(feats.todense()))
        lbs = torch.LongTensor(np.where(lbs)[1])
        flags = torch.LongTensor(np.where(flags)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense().unsqueeze(-1)

        return adj, feats, lbs, flags

    # custom data loader
    def collate_fn(input_batch):
        # data
        adj, feats, lbs, flags = calculate_adjacency_matrix(input_batch)

        # train, val, test split (For approved samples: train:val:test = 60%:20%:20%)
        idx_default = torch.nonzero(lbs == 1).squeeze()
        idx_non_default = torch.nonzero(lbs == 0).squeeze()
        indices = torch.cat((idx_default, idx_non_default), dim=0)
        # train
        num_approved = indices.shape[0]
        indices = indices[torch.randperm(num_approved)]  # Shuffle
        index_train_labeled = indices[: int(0.6 * num_approved)]
        index_train_unlabeled = torch.nonzero(lbs == 2).squeeze()
        index_train = torch.LongTensor(torch.cat((index_train_labeled, index_train_unlabeled), dim=0))
        # val
        index_val = torch.LongTensor(indices[int(0.6 * num_approved): int(0.8 * num_approved)])
        # test
        index_test = torch.LongTensor(indices[int(0.8 * num_approved):])

        return adj, feats, lbs, index_train, index_train_labeled, index_train_unlabeled, index_val, index_test

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)

    processed_batches = []
    for idx, BATCH_SIZE in enumerate(dataloader):
        adj, features, labels, idx_train, idx_train_labeled, idx_train_unlabeled, idx_val, idx_test = BATCH_SIZE
        # add to list
        processed_data = {
            'adj': adj,
            'features': features,
            'labels': labels,
            'idx_train': idx_train,
            'idx_train_labeled': idx_train_labeled,
            'idx_train_unlabeled': idx_train_unlabeled,
            'idx_val': idx_val,
            'idx_test': idx_test
        }
        processed_batches.append(processed_data)
    return processed_batches


if __name__ == "__main__":
    load_LendingClub(dataset_name='Lending1')
