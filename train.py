import argparse
import os
import random
import numpy as np
import torch
from torch import optim, nn
from torchmetrics import F1Score, AUROC
from os.path import join
from utils.model import *
from utils.logger import Logger
from utils.metric import *
from utils.dataset import Dataset
from configs.loader import load_yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SEED = 42


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_loop(model, train_iter, val_iter, optimizer, loss, epochs, device, add_weights_file, log_file):
    log_training_loss = []
    log_training_accuracy = []
    log_val_accuracy = []
    log_f1_scores = []
    log_auc_scores = []
    best_val_acc = 0.0

    f1_metric = F1Score(task="binary", num_classes=2).to(device)
    auc_metric = AUROC(task="binary", num_classes=2).to(device)

    model.to(device)
    with open(log_file, 'w') as f:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, acc_sum, samples_sum = 0.0, 0.0, 0
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                samples_num = X.shape[0]

                # Forward pass
                output = model(X)
                log_softmax_output = torch.log(output)
                l = loss(log_softmax_output, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                # Metrics calculation
                loss_sum += l.item() * samples_num
                acc_sum += calculate_accuracy(output, y) * samples_num
                samples_sum += samples_num
            train_acc = acc_sum / samples_sum

            # Validation metrics
            val_acc, val_f1, val_auc = evaluate(model, val_iter, device, f1_metric, auc_metric)

            if val_acc >= best_val_acc:
                save_hint = "save the model to {}".format(add_weights_file)
                torch.save(model.state_dict(), add_weights_file)
                best_val_acc = val_acc
            else:
                save_hint = ""

            # Log results to file
            log_message = f"epoch:{epoch}, loss:{loss_sum / samples_sum:.4f}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, best_record:{best_val_acc:.4f}, F1:{val_f1:.4f}, AUC:{val_auc:.4f}  {save_hint}\n"
            f.write(log_message)

            # Append metrics to lists for plotting
            log_training_loss.append(loss_sum / samples_sum)
            log_training_accuracy.append(train_acc)
            log_val_accuracy.append(val_acc)
            log_f1_scores.append(val_f1)
            log_auc_scores.append(val_auc)

    log = {
        "loss": log_training_loss,
        "acc_train": log_training_accuracy,
        "acc_val": log_val_accuracy,
        "f1": log_f1_scores,
        "auc": log_auc_scores
    }
    return log


def evaluate(model, val_iter, device, f1_metric, auc_metric):
    model.eval()
    acc_sum, samples_sum = 0.0, 0
    f1_metric.reset()
    auc_metric.reset()

    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device)
            output = model(X)

            preds = torch.argmax(output, dim=1)
            acc_sum += calculate_accuracy(output, y) * X.size(0)
            samples_sum += X.size(0)
            f1_metric.update(preds, y)
            auc_metric.update(preds, y)

    val_acc = acc_sum / samples_sum
    val_f1 = f1_metric.compute().item()
    val_auc = auc_metric.compute().item()
    return val_acc, val_f1, val_auc


def plot_logs(log, save_path="training_logs.png"):
    epochs = range(1, len(log["loss"]) + 1)
    plt.figure()
    plt.plot(epochs, log["loss"], label="Training Loss")
    plt.plot(epochs, log["acc_train"], label="Training Accuracy")
    plt.plot(epochs, log["acc_val"], label="Validation Accuracy")
    plt.plot(epochs, log["f1"], label="F1 Score")
    plt.plot(epochs, log["auc"], label="AUC Score")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Training and Validation Metrics")
    plt.savefig(save_path)


def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            output = model(X)
            features.append(output.cpu().numpy())
            labels.append(y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def plot_tsne(features, labels, num_classes, save_path="tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    plt.figure(figsize=(10, 8))
    for class_idx in range(num_classes):
        idxs = (labels == class_idx)
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], color=colors[class_idx], label=f"Class {class_idx}",
                    alpha=0.6)

    plt.legend()
    plt.title("t-SNE Feature Distribution")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_path)


def main(args):
    set_random_seed(SEED)

    if_gpu = args.gpu
    dataset_name = args.dataset
    dataset_level = args.level
    branch_selection = args.branch

    args_model = load_yaml("configs/args_model.yaml")
    args_train = load_yaml("configs/args_train.yaml")

    BLOCK_SIZE = args_train["BLOCK_SIZE"]
    BATCH_SIZE = args_train["BATCH_SIZE"]

    add_weights = args_train["add_weights"]
    if not os.path.exists(add_weights):
        os.makedirs(add_weights)

    EPOCHS_g1 = args_train["EPOCHS_g1"]
    LEARNING_RATE_g1 = args_train["LEARNING_RATE_g1"]
    weights_name_g1 = args_train["weights_name_g1"]

    EPOCHS_g2 = args_train["EPOCHS_g2"]
    LEARNING_RATE_g2 = args_train["LEARNING_RATE_g2"]
    weights_name_g2 = args_train["weights_name_g2"]

    device = "cuda" if if_gpu and torch.cuda.is_available() else "cpu"

    dataset = Dataset(add_root=args_train["add_dataset_root"], name=dataset_name, level=dataset_level)

    logger = Logger()
    logger.register_status(dataset=dataset, device=device, branch_selection=branch_selection)
    logger.register_args(**args_train, **args_model)
    logger.print_logs_training()

    train_iter_A = None
    train_iter_B = None
    val_iter_A = None
    val_iter_B = None
    if branch_selection == 'g1':
        train_iter_A = dataset.load_data_train_g1(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A = dataset.load_data_val_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        train_iter_B = dataset.load_data_train_g2(BLOCK_SIZE, BATCH_SIZE)
        val_iter_B = dataset.load_data_val_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        train_iter_A, train_iter_B = dataset.load_data_train_all(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A, val_iter_B = dataset.load_data_val_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    if branch_selection == 'g1' or branch_selection == 'all':
        assert train_iter_A and val_iter_A
        g1 = LRNet(**args_model)
        optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE_g1)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g1)
        log_file_g1 = "training_log_g1_c23.txt"
        log_g1 = train_loop(g1, train_iter_A, val_iter_A, optimizer, loss, EPOCHS_g1, device, add_weights_file,
                            log_file_g1)
        plot_logs(log_g1, "log_g1_c23.png")

        features, labels = extract_features(g1, val_iter_A, device)
        plot_tsne(features, labels, num_classes=2, save_path="tsne_g1_c23.png")

    if branch_selection == 'g2' or branch_selection == 'all':
        assert train_iter_B and val_iter_B
        g2 = LRNet(**args_model)
        optimizer = optim.Adam(g2.parameters(), lr=LEARNING_RATE_g2)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g2)
        log_file_g2 = "training_log_g2_c23.txt"
        log_g2 = train_loop(g2, train_iter_B, val_iter_B, optimizer, loss, EPOCHS_g2, device, add_weights_file,
                            log_file_g2)
        plot_logs(log_g2, "log_g2_c23.png")

        features, labels = extract_features(g2, val_iter_B, device)
        plot_tsne(features, labels, num_classes=2, save_path="tsne_g2_c23.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training codes of LRNet (PyTorch version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true', help="If use the GPU(CUDA) for training.")
    parser.add_argument('-d', '--dataset', type=str, choices=['DF', 'F2F', 'FS', 'NT', 'FF_all'], default='FF_all',
                        help="Select the dataset used for training.")
    parser.add_argument('-l', '--level', type=str, choices=['raw', 'c23', 'c40'], default='c23',
                        help="Select the dataset compression level.")
    parser.add_argument('-b', '--branch', type=str, choices=['g1', 'g2', 'all'], default='all',
                        help="Select which branch of the LRNet to be trained.")
    args = parser.parse_args()
    main(args)
