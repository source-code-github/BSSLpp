
import numpy as np
from itertools import chain
import copy

from sklearn.cluster import KMeans


class BSSLpp():
    '''
    Args:
        base_estimator: a base learner used in boosting
        Tk: a number generated base learners
        n_classes: a number of classes
        len_lab: a total number of labeled data points
    '''
    def __init__(self, base_estimator, Tk, n_classes, len_lab):
        super().__init__()
        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.Tk = Tk
        self.len_lab = len_lab
        self.Betas = []
        self.composites = []
        self.betas_inside_composites = []
        self.k = 0

    def classify_final_ensemble(self, data):
        '''
        Args:
            data: the set of samples to predict the labels for.
        
        Return:
            predictions: predictions for all samples in data

        '''

        Betas = np.array(list(chain.from_iterable(self.Betas)))
        Weights = np.log(1./Betas)

        p = np.zeros((len(data), self.n_classes))

        for k in range(len(self.composites)):
            ens_preds = self.classify_ensemble(
                data, self.composites[k], self.betas_inside_composites[k])

            for m in range(len(ens_preds)):
                p[m, ens_preds[m]] = p[m, ens_preds[m]] + Weights[k]

        predictions = np.argmax(p, axis=1)

        return predictions

    def classify_ensemble(self, data, classifiers, betas):
        '''
        Args:
            data: the set of samples to predict the labels for.
            classifiers: base classifers used to form an ensemble.
            betas: the weight coefficients of each classifer in classifiers
        Return:
            predictions: predictions for all samples in data obtained from ensemble of classifers

        '''
        beta = np.array(betas)
        weights = np.log(1./beta)
        p = np.zeros((len(data), self.n_classes))

        for k in range(len(classifiers)):

            pred = classifiers[k].predict(data)

            for m in range(len(pred)):
                p[m, int(pred[m])] = p[m, int(pred[m])] + weights[k]

        predictions = np.argmax(p, axis=1)
        return predictions

    def predict(self, data):
        '''
        Args:
            data: the set of samples to predict the labels for.
        
        Return:
            predictions: predictions for all samples in data

        '''
        predictions = self.classify_final_ensemble(data)
        return predictions

    def score(self, data, labels):
        '''
        Args:
            data: the set of samples to predict the labels for.
            labels: the true labels of samples in data
        Return:
            accuracy: the accuracy of predictions with respect to labels

        '''
        predictions = self.predict(data)
        accuracy = sum((np.equal(predictions, labels)).astype(int)) / len(labels)

        return accuracy

    def fit_batch(self, data_train_k, labels_train_k):
        ''' generate an AdaBoost-like ensemble using data_train_k and labels_train_k

        Args:
            data_train_k: the set of samples to predict the labels for.
            labels_train_k: the true labels of samples in data
        '''

        # initialize sample weights
        betta = 0.9
        D_l = np.ones(self.len_lab) * betta / self.len_lab
        D_u = np.ones(
            len(labels_train_k)-self.len_lab) * (1 - betta) / (len(labels_train_k)-self.len_lab)
        D = np.concatenate((D_l, D_u))

        classifiers = []
        betas = []
        Betas = []
        t = 0
        while t < self.Tk:
            
            # normalize sample weights
            D = D / sum(D)

            # use D to randomly draw data_train_k_sample of size len(D)*0.5 from data_train_k

            index = np.random.choice(a=len(D), size=int(len(D)*0.5), replace=False, p=D)
            data_train_k_sample = data_train_k[index]
            labels_train_k_sample = labels_train_k[index]

            h_t = copy.deepcopy(self.base_estimator)

            # train a base classifer
            h_t.fit(data_train_k_sample, labels_train_k_sample)

            classifiers.append(h_t)

            h_t_predictions = h_t.predict(data_train_k)

            # compute the error of base classifier
            epsilon_kt = sum(D * (np.not_equal(h_t_predictions, labels_train_k)).astype(int))

            # compute the normalized error of base classifier
            beta = (epsilon_kt+1e-8)/(1-epsilon_kt+1e-8)
            betas.append(beta)

            # if epsilon_kt is worse than random guessing, stop
            if round(epsilon_kt, 2) > 1.0 - (1.0 / self.n_classes):
                classifiers.pop()
                betas.pop()
                break
            
            # construct weighted majority classifer (AdaBoost-like ensemble) and use to predict the labels of data_train_k
            H_t_predictions = self.classify_ensemble(data_train_k, classifiers, betas)

            # compute the error of weighted majority classifer (AdaBoost-like ensemble)
            E_kt = sum(D * (np.not_equal(H_t_predictions, labels_train_k)).astype(int))

            # if E_kt is worse than random guessing, stop
            if round(E_kt, 2) > 1.0 - (1.0 / self.n_classes):
                classifiers.pop()
                betas.pop()
                break
            
            # compute the normalized error of weighted majority classifer (AdaBoost-like ensemble)
            Bkt = (E_kt+1e-8) / (1 - E_kt+1e-8)

            # update the sample weights
            D[H_t_predictions == labels_train_k] = Bkt * D[H_t_predictions == labels_train_k]

            Betas.append(Bkt)

            t = t + 1

            append_classifiers = np.array(classifiers)
            self.composites.append(append_classifiers)
            append_betas = np.array(betas)
            self.betas_inside_composites.append(append_betas)

        self.k = self.k + 1
        self.Betas.append(Betas)

    def fit(self, X_lab, y_lab, X_unl_all):
        ''' generate an ensemble of AdaBoost-like ensemble classifiers using X_lab, y_lab, and X_unl_all
        
        Args:
            X_lab: the set of labeled samples
            y_lab: the set labels of labeled samples
            X_unl_all: the set of unlabeled samples
        '''

        count = 0
        flag = True

        X_train = np.copy(X_lab)
        y_train = np.copy(y_lab)

        X_unl_rest = np.copy(X_unl_all)

        while flag == True:
            count += 1

            # select and pseudo-label nearest unlabeled samples
            X_unl, pseudo_y_unl, rest_unl_indices = self.PLNS(X_train, y_train, X_unl_rest, count)
            
            if len(rest_unl_indices) == 0:
                # if no unlabeled samples are left, stop while loop
                flag = False
            else:
                X_unl_rest = X_unl_rest[rest_unl_indices]

            # add pseudo-labeled unlabeled samples to current training set
            X_train = np.concatenate((X_train, X_unl))
            y_train = np.concatenate((y_train, pseudo_y_unl))

            # generate an AdaBoost-like ensemble classifier
            self.fit_batch(X_train, y_train)

            # update the pseudo-labels of unlabeled samples
            pseudo_y_unl = self.predict(X_train[self.len_lab:])
            y_train[self.len_lab:] = pseudo_y_unl

    def PLNS(self, X_train, y_train, X_unl_rest, count):
        ''' select and pseduo-label nearest unlabeled samples for each labeled sample 
        
        Args:
            X_train: the set of training samples
            y_train: the set labels of training samples
            X_unl_rest: the set of unlabeled samples
            count: current iteration (counter)
        
        Return:
            X_unl: the selected unlabeled samples
            pseudo_y_unl: the pseudo-labels of X_unl
            rest_unl_indices: the indices of remaining unlabeled samples
        '''

        # get indices of nearest and remaining unlabeled samples
        nn_indices, rest_unl_indices = self.get_NN_indices(X_train, X_unl_rest)

        # select nearest unlabeled samples using nn_indices
        X_unl = X_unl_rest[nn_indices]

        if count == 1:
            # X_train includes only labeled samples, use centriods of X_train
            unique_labels = np.unique(y_train)
            lab_centroids = []
            for label in unique_labels:
                cluster_lab_data = X_train[y_train == label]
                lab_centroid = np.mean(cluster_lab_data, axis=0)
                lab_centroids.append(lab_centroid)
            lab_centroids = np.array(lab_centroids)

            pseudo_y_unl = []
            for x_unl in X_unl:
                euc_dist = np.sqrt(np.sum((x_unl - lab_centroids)**2, axis=1))
                y_cluster = np.argmin(euc_dist)
                pseudo_y_unl.append(y_cluster)
        
        else:
            # initialize centriods using k-means++
            kmeans = KMeans(n_clusters=self.n_classes, random_state=42,
                            init='k-means++', n_init='auto')

            kmeans.fit(X_train,)
            unl_cluster_idx = kmeans.predict(X_unl)

            # for convience convert each cluster index "0" to "cluster_{idx}" format, refer as "cluster name"
            cluster_names = [f'cluster_{cluster_id}' for cluster_id in kmeans.labels_]
            unl_cluster_names = [f'cluster_{cluster_id}' for cluster_id in unl_cluster_idx]

            # assign class labels to each cluster based on weighted majority of true labeled samples in each cluster 
            cluster_labels = self.clusterToLabel(
                y_lab=y_train[: self.len_lab],
                cluster_names=cluster_names[: self.len_lab])

            # assign pseudo-labels based on assigned class labels of clusters
            pseudo_y_unl = [int(cluster_labels[cluster_name]) for cluster_name in unl_cluster_names]

        return X_unl, pseudo_y_unl, rest_unl_indices

    def get_NN_indices(self, X_train, X_unl_rest):
        ''' get the indices of nearest unlabeled samples for each labeled sample 
        
        Args:
            X_train: the set of training samples
            X_unl_rest: the set of unlabeled samples
        
        Return:
            nn_indices: the indices of selected unlabeled samples
            rest_unl_indices: the indices of remaining unlabeled samples
        '''

        # calculate distances between samples X_train and X_unl_rest
        ord = 1
        Dist_lu = np.ones((len(X_train), len(X_unl_rest)))
        for i in range(len(X_train)):
            for j in range(len(X_unl_rest)):
                dist_ = np.linalg.norm(X_train[i]-X_unl_rest[j], ord=ord)
                Dist_lu[i, j] = dist_

        # get indices of the nearest unlabeled sample for each sample in X_train 
        nn_indices = set()
        for unls_dist in Dist_lu:
            unls_dist_sorted = np.sort(unls_dist)
            nearest_dist = unls_dist_sorted[0]
            for i, unl_dist in enumerate(unls_dist):
                if unl_dist == nearest_dist:
                    nn_indices.add(i)

        nn_indices = np.array(list(nn_indices))

        # get indices of remaining unlabeled samples
        rest_unl_indices = np.array([i for i in range(len(X_unl_rest)) if i not in nn_indices])

        return nn_indices, rest_unl_indices

    def clusterToLabel(self, y_lab, cluster_names):
        ''' assign class labels to generated clusters based on weighted majority of true labeled samples in each cluster 
        
        Args:
            y_lab: labels of labeled samples in clusters
            cluster_names: cluster names of labeled samples in clusters
        
        Return:
            cluster_labels: the assigned labels for each cluster in the form of dictionary
        '''

        cluster_labels = {}
        cluster_counts = {}

        all_cluster_names = [f'cluster_{i}' for i in range(self.n_classes)]
        all_labels = np.arange(self.n_classes)

        for cluster in all_cluster_names:
            cluster_counts[cluster] = {}
            for label in all_labels:
                cluster_counts[cluster][str(label)] = 0

        # count the number of labeled samples from each class in each cluster
        for label, cluster in zip(y_lab, cluster_names):
            cluster_counts[cluster][str(label)] += 1

        # divide the number of labeled samples from each class in each cluster by the total number of labeled samples in each class
        for cluster in all_cluster_names:
            for label in all_labels:
                for i in range(self.n_classes):
                    if label == i:
                        cluster_counts[cluster][str(label)] /= np.count_nonzero(y_lab == i)

        # assign to each cluster the class label by weighted majority of true labeled samples in each cluster
        for cluster, counts in cluster_counts.items():
            cluster_labels[cluster] = max(counts, key=counts.get)

        return cluster_labels
