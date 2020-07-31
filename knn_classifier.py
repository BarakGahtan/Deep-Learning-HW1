import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.n_classes = len(set(y_train.numpy()))
        self.x_train = x_train
        self.y_train = y_train

        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            top_indices = torch.topk(torch.t(dist_matrix)[i], self.k, largest=False)[1]
            y_pred[i] = torch.mode(self.y_train[top_indices])[0]
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """
    dists = None
    #x1 is train, x2 is test
    # ====== YOUR CODE: ======
    size_x1 = x1.size(0)
    size_x2 = x2.size(0)
    x1_norm = torch.sum(x1 ** 2,dim = 1, dtype=torch.float,keepdim=True).expand(size_x1,size_x2)
    x2_norm = torch.sum(x2 ** 2,dim = 1, dtype=torch.float,keepdim=True).transpose(0, 1).expand(size_x1,size_x2)
    
    x2_transpose = x2.transpose(0, 1)

    x1_x2 = -2 * torch.mm(x1, x2_transpose)
    dist_pow2 = x1_x2 + x1_norm + x2_norm
    dists = torch.sqrt(dist_pow2)
    # ========================
    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    diff = y - y_pred
    ones = torch.ones(y.shape)
    diff = torch.where(diff != 0, ones, diff)
    accuracy = 1 - ((torch.sum(diff, 0).item()) / (y.shape[0]))
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        validation_ratio = 1 / num_folds
        k_accuracies = []
        for j in range(num_folds):
            model_to_train, model_to_val = dataloaders.create_train_validation_loaders(ds_train, validation_ratio)
            model = model.train(model_to_train)
            x_val, y_val = dataloader_utils.flatten(model_to_val)
            y_pred = model.predict(x_val)
            dist = accuracy(y_val, y_pred)
            k_accuracies.append(dist)
        accuracies.append(k_accuracies)
        # ========================
    best_k = torch.mean(torch.tensor(accuracies),dim=1)
    best_k_idx = torch.argmax(best_k).item()
    #best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies


