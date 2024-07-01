from utils import load_LendingClub, kolmogorov_smirnov
from model import LatentRelationExploration, target_distribution, classifier
from IB_loss import IBLoss

import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(dataset):
    acc_list = []  # ACC
    auc_list = []  # AUC
    ks_list = []  # KS
    ib_loss_list = []  # ib_loss
    clu_loss_list = []  # clu_loss

    for idx, batch in enumerate(dataset):
        adj = batch['adj']
        features = batch['features']
        labels = batch['labels']
        idx_train = batch['idx_train']
        idx_train_labeled = batch['idx_train_labeled']
        idx_val = batch['idx_val']
        idx_test = batch['idx_test']
        n_classes = int(labels.max() + 1)

        latent_relation_model = LatentRelationExploration(in_features=features.shape[1],
                                                          n_hidden_gcn=args.n_hidden_gcn,
                                                          n_classes=n_classes,
                                                          n_heads=args.n_heads,
                                                          n_layers=args.n_layers,
                                                          is_concat=False,
                                                          share_weights=args.share_weights)

        # loss
        criterion_ib = IBLoss(temp=0.1)
        # optimizer
        optimizer = optim.Adam(latent_relation_model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # Cuda
        if args.cuda:
            latent_relation_model.cuda()
            features = features.cuda()
            adj = adj.cuda().squeeze(-1)
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        # Initiate parameters of the clustering layer
        features_gcn, latent_gcn, q = latent_relation_model(features, adj)
        kmeans = KMeans(n_clusters=args.n_clusters)
        _ = kmeans.fit_predict(features_gcn.detach().cpu().numpy())  # numpy.ndarray
        latent_relation_model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)  # shape=(n_clusters, n_hidden_gcn)

        # Train
        latent_relation_model.train()
        acc_clu = 0
        bad_counter = 0
        best = args.epochs + 1

        """
        Step1: Unsupervised Training
        """

        for epoch in range(args.epochs):

            # training interval
            if epoch % args.update_interval == 0:
                _, _, tmp_q = latent_relation_model(features, adj)
                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            acc = accuracy_score(np.array(labels[idx_train_labeled].cpu()), y_pred[idx_train_labeled])
            acc_list.append(acc.item())

            if acc.item() > acc_clu:
                acc_clu = acc.item()
                bad_counter = 0
                torch.save(latent_relation_model.state_dict(), args.model_path)
            else:
                bad_counter += 1

            # loss
            features_gcn, latent_gcn, q = latent_relation_model(features, adj)
            clu_loss = F.kl_div(q.log(), p)
            ib_loss = criterion_ib(features_gcn, latent_gcn, y_pred, r_negative=args.neg_ratio)
            loss_un = args.clu_weight * clu_loss + args.ib_weight * ib_loss
            # record
            ib_loss_list.append(ib_loss.item())
            clu_loss_list.append(clu_loss.item())

            # back propagation
            optimizer.zero_grad()
            loss_un.backward()
            optimizer.step()

            # early stop
            if acc_clu > 0.8:
                break

        """
        Step2: Supervised Training
        """
        criterion = nn.CrossEntropyLoss()
        checkpoint = torch.load(args.model_path)
        latent_relation_model.load_state_dict(checkpoint)
        features_gcn, latent_gcn, _ = latent_relation_model(features, adj)
        latent_embeds = (features_gcn + latent_gcn).detach()

        aucs = []
        kss = []

        for i in range(50):
            auc_max = 0
            ks_max = 0

            classification = classifier(latent_embeds.shape[1], n_classes).cuda()
            opt = torch.optim.Adam(classification.parameters(), lr=5e-3, weight_decay=args.weight_decay)

            for _ in range(700):
                classification.train()
                opt.zero_grad()
                logits = classification(latent_embeds)
                loss_ce = criterion(logits[idx_train_labeled], labels[idx_train_labeled])
                loss_ce.backward()
                opt.step()
                preds = torch.argmax(classification(latent_embeds), dim=1)

                # Evaluation
                auc = roc_auc_score(labels[idx_test].cpu().detach().numpy(),
                                    preds[idx_test].cpu().detach().numpy())
                ks = kolmogorov_smirnov(labels[idx_test].cpu().detach().numpy(),
                                        preds[idx_test].cpu().detach().numpy())

                if auc > auc_max:
                    auc_max = auc
                    ks_max = ks

            aucs.append(auc_max * 100)
            kss.append(ks_max * 100)

        # For each batch: Print the mean and standard deviation of metrics
        print(
            'batch {}, AUC Mean:{:.2f}%, AUC Std:{:.2f}%, KS Mean:{:.2f}%, KS Std:{:.2f}%'.format(
                (idx + 1),
                torch.tensor(aucs).mean().item(),
                torch.tensor(aucs).std().item(),
                torch.tensor(kss).mean().item(),
                torch.tensor(kss).std().item()
            )
        )

        # For the dataset: Print the mean and standard deviation of metrics
        auc_list.append(torch.tensor(aucs).mean().item())
        ks_list.append(torch.tensor(kss).mean().item())
    print(
        '{}, AUC Mean:{:.2f}%, AUC Std:{:.2f}%, KS Mean:{:.2f}%, KS Std:{:.2f}%'.format(
            args.dataset_name,
            torch.tensor(auc_list).mean().item(),
            torch.tensor(auc_list).std().item(),
            torch.tensor(ks_list).mean().item(),
            torch.tensor(ks_list).std().item()
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--fast_mode', action='store_true', default=True)
    parser.add_argument('--sparse', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_hidden_gcn', default=32, type=int)
    parser.add_argument('--n_heads', default=2, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_clusters', default=2, type=int)
    parser.add_argument('--clu_weight', default=1e-3, type=float)
    parser.add_argument('--ib_weight', default=1, type=float)
    parser.add_argument('--update_interval', default=2, type=int)
    parser.add_argument('--neg_ratio', default=0.1, type=float)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--share_weights', action='store_true', default=True)
    parser.add_argument('--dataset_name', type=str, default='')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.init()
    device = torch.device("cuda:0")

    # Set the dataset
    args.dataset_name = 'Lending1'
    # Set the desired model path
    args.model_path = os.path.join('best_model_updated', args.dataset_name + '_best_model.pkl')
    # Parameters
    if args.dataset_name == 'Lending1':
        dataset = load_LendingClub(dataset_name=args.dataset_name, p=0.95, BATCH_SIZE=1000)
        args.lr = 5e-3
        args.weight_decay = 1e-5
        args.n_hidden_gcn = 64
        args.n_heads = 8
        args.n_layers = 2
        args.clu_weight = 1e-3
        args.ib_weight = 1
        args.update_interval = 2
        args.neg_ratio = 0.7
        # Training model
        train(dataset)
    elif args.dataset_name == 'Lending2':
        dataset = load_LendingClub(dataset_name=args.dataset_name, p=0.9, BATCH_SIZE=1000)
        args.lr = 5e-3
        args.weight_decay = 1e-5
        args.n_hidden_gcn = 64
        args.n_heads = 8
        args.n_layers = 2
        args.clu_weight = 1e-3
        args.ib_weight = 1
        args.update_interval = 2
        args.neg_ratio = 0.7
        # Training model
        train(dataset)
    elif args.dataset_name == 'Lending3':
        dataset = load_LendingClub(dataset_name=args.dataset_name, p=0.9, BATCH_SIZE=1000)
        args.lr = 5e-3
        args.weight_decay = 5e-4
        args.n_hidden_gcn = 64
        args.n_heads = 20
        args.n_layers = 1
        args.clu_weight = 1e-3
        args.ib_weight = 1
        args.update_interval = 2
        args.neg_ratio = 0.1
        # Training model
        train(dataset)
    else:
        raise ValueError(f'Dataset {args.dataset_name} not found.')
