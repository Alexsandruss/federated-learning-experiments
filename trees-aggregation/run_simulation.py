import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from math import ceil, floor
import logging


class BaseWorkload:
    def __init__(self, dataset, n_nodes, split='KMeans'):
        np.random.seed(42)
        self.dataset = dataset
        self.n_nodes = n_nodes
        self.split = split
        self.all_data = None
        self.test_data = None
        self.global_metrics = {}
        self.local_metrics = {}
        self.local_subsets = {i: None for i in range(self.n_nodes)}
        self.local_models = {i: None for i in range(self.n_nodes)}
        self.n_rounds = 0

    def kmeans_split(self, x, y):
        x_norm = (x - x.mean(axis=0)) / x.std(axis=0)
        x_norm = x_norm[:, (np.isnan(x_norm).sum(axis=0) + np.isinf(x_norm).sum(axis=0)) == 0]
        kmeans = KMeans(n_clusters=self.n_nodes, random_state=np.random.randint(2 ** 31 - 1))
        return kmeans.fit_predict(x_norm)

    def random_split(self, x, y):
        return np.random.randint(0, self.n_nodes, size=(x.shape[0], ))

    def load_and_split_dataset(self):
        x_train, y_train = np.load(f'data/{self.dataset}_x_train.npy', allow_pickle=True), np.load(f'data/{self.dataset}_y_train.npy', allow_pickle=True).astype(np.int32)
        x_test, y_test = np.load(f'data/{self.dataset}_x_test.npy', allow_pickle=True), np.load(f'data/{self.dataset}_y_test.npy', allow_pickle=True).astype(np.int32)
        y_train, y_test = y_train.reshape((y_train.shape[0], )), y_test.reshape((y_test.shape[0], ))
        logging.info(f'{self.dataset} dataset is loading with {self.split} split method')
        logging.info(f'Training data shape: {x_train.shape}')
        logging.info(f'Testing data shape: {x_test.shape}')
        self.all_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        if self.split == 'KMeans':
            split_indices = self.kmeans_split(x_train, y_train)
        elif self.split == 'Random':
            split_indices = self.random_split(x_train, y_train)
        else:
            raise NotImplementedError('Unknown split method')
        for i in range(self.n_nodes):
            select_indices = (split_indices == i)
            logging.info(f'{select_indices.sum()} samples for {i} node, bincount {np.bincount(y_train[select_indices])}')
            self.local_subsets[i] = (x_train[select_indices], y_train[select_indices])

    def local_step(self, i):
        raise NotImplementedError

    def aggregation_step(self, round):
        raise NotImplementedError

    def run_round(self, round):
        for i in range(self.n_nodes):
            t0 = timer()
            self.local_step(i)
            t1 = timer()
            logging.debug(f'Local step {round} on node {i} time[s]: {t1 - t0}')
        t0 = timer()
        self.aggregation_step(round)
        t1 = timer()
        logging.debug(f'Aggregation step {round} time[s]: {t1 - t0}')
        self.n_rounds += 1

    def run(self, n_rounds):
        self.load_and_split_dataset()
        for i in range(n_rounds):
            self.run_round(i)

    def save_metrics(self):
        raise NotImplementedError


class FeatureImportanceDistanceWorkload(BaseWorkload):
    def init(self, n_estimators_per_node_per_round, m_estimators_to_select_per_round, method='kmeans', method_params=None):
        self.n_estimators_per_node_per_round = n_estimators_per_node_per_round
        self.m_estimators_to_select_per_round = m_estimators_to_select_per_round
        self.method = method
        self.method_params = method_params
        self.local_metrics = {
            'Balanced accuracy': {i: [] for i in range(self.n_nodes)},
            'Accuracy': {i: [] for i in range(self.n_nodes)},
            'Precision': {i: [] for i in range(self.n_nodes)},
            'Recall': {i: [] for i in range(self.n_nodes)}
        }
        self.global_metrics = {
            'Balanced accuracy': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': []
        }

    def local_step(self, i):
        x, y = self.local_subsets[i]
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators_per_node_per_round,
            n_jobs=cpu_count(),
            random_state=np.random.randint(2 ** 31 - 1))
        rf.fit(x, y)
        # for i in range(self.n_estimators_per_node_per_round):
        #     rf.estimators_[i].feature_importances_ = permutation_importance(rf.estimators_[i], x, y, scoring='neg_log_loss', n_repeats=8)
        if self.local_models[i] is None:
            self.local_models[i] = rf
        else:
            self.local_models[i].estimators_ += rf.estimators_
        self.local_models[i].n_estimators = len(self.local_models[i].estimators_)

    def aggregation_step(self, round):
        all_trees = self.local_models[0].estimators_[:len(self.local_models[0].estimators_) - self.n_estimators_per_node_per_round]
        for i in self.local_models.keys():
            all_trees += self.local_models[i].estimators_[len(self.local_models[i].estimators_) - self.n_estimators_per_node_per_round:]

        if self.method == 'Random':
            aggregated_model = self.random_aggregation(round, all_trees, self.method_params)
        elif self.method == 'KMeans':
            aggregated_model = self.kmeans_aggregation(round, all_trees, self.method_params)
        elif self.method == 'Center':
            aggregated_model = self.center_aggregation(round, all_trees, self.method_params)
        else:
            raise NotImplementedError('Unknown aggregation method')
        for i in self.local_models.keys():
            self.local_models[i] = deepcopy(aggregated_model)

        # estimate model on all data
        x_test, y_test = self.test_data
        y_pred = aggregated_model.predict(x_test)
        self.global_metrics['Balanced accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        self.global_metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        self.global_metrics['Precision'].append(precision_score(y_test, y_pred))
        self.global_metrics['Recall'].append(recall_score(y_test, y_pred))
        # estimate model on node subsets
        for i in range(self.n_nodes):
            x_local, y_local = self.local_subsets[i]
            y_pred = self.local_models[i].predict(x_local)
            self.local_metrics['Balanced accuracy'][i].append(balanced_accuracy_score(y_local, y_pred))
            self.local_metrics['Accuracy'][i].append(accuracy_score(y_local, y_pred))
            self.local_metrics['Precision'][i].append(precision_score(y_local, y_pred))
            self.local_metrics['Recall'][i].append(recall_score(y_local, y_pred))

    def random_aggregation(self, round, all_trees, params):
        np.random.shuffle(all_trees)
        aggregated_model = deepcopy(self.local_models[0])
        aggregated_model.estimators_ = all_trees[:self.m_estimators_to_select_per_round * (round + 1)]
        aggregated_model.n_estimators = len(aggregated_model.estimators_)
        return aggregated_model

    def get_feature_importances(self, all_trees):
        feature_importances = np.array([tree.feature_importances_ for tree in all_trees])
        feature_importances_norm = (feature_importances - feature_importances.mean(axis=0)) / feature_importances.std(axis=0)
        feature_importances_norm = feature_importances_norm[:, (
            np.isnan(feature_importances_norm).sum(axis=0) + np.isinf(feature_importances_norm).sum(axis=0)) == 0]
        return pd.DataFrame(feature_importances), pd.DataFrame(feature_importances_norm)

    def kmeans_aggregation(self, round, all_trees, params):
        feature_importances, feature_importances_norm = self.get_feature_importances(all_trees)

        kmeans = KMeans(n_clusters=self.n_nodes, random_state=np.random.randint(2 ** 31 - 1))
        fi_cluster_indices = kmeans.fit_predict(feature_importances_norm)
        fi_cluster_centers = kmeans.cluster_centers_

        aggregated_model = deepcopy(self.local_models[0])
        aggregated_model.estimators_ = []
        n_trees_to_aggregate = self.m_estimators_to_select_per_round * (round + 1)
        for i in range(self.n_nodes):
            n_trees_to_select = floor(((fi_cluster_indices == i).sum() / len(all_trees)) * n_trees_to_aggregate)
            cluster_indices = feature_importances.index[fi_cluster_indices == i]
            cluster_fis = feature_importances_norm.iloc[cluster_indices]
            distances_to_center = ((cluster_fis - fi_cluster_centers[i]) ** 2).sum(axis=1).sort_values(ascending=False)
            for j in range(n_trees_to_select):
                 aggregated_model.estimators_.append(all_trees[distances_to_center.index[j]])
        aggregated_model.n_estimators = len(aggregated_model.estimators_)
        return aggregated_model

    def center_aggregation(self, round, all_trees, params):
        feature_importances, feature_importances_norm = self.get_feature_importances(all_trees)

        feature_importances_center = np.average(feature_importances_norm, axis=0)
        distances = pd.Series(((feature_importances_norm - feature_importances_center) ** 2).sum(axis=1)).sort_values(ascending=False)

        aggregated_model = deepcopy(self.local_models[0])
        aggregated_model.estimators_ = []
        for i in range(self.m_estimators_to_select_per_round * (round + 1)):
            aggregated_model.estimators_.append(all_trees[distances.index[i]])
        aggregated_model.n_estimators = len(aggregated_model.estimators_)
        return aggregated_model

    def save_metrics(self):
        for metric in self.local_metrics.keys():
            # make graphics
            fig, ax = plt.subplots()
            fig.set_size_inches(16, 9)
            for i in range(self.n_nodes):
                ax.plot([i for i in range(self.n_rounds)], self.local_metrics[metric][i], label=f'node {i}', marker='o')
            ax.plot([i for i in range(self.n_rounds)], self.global_metrics[metric], label='global', marker='o', color='black')
            ax.set(xlabel='Round', ylabel='Metric', title=metric)
            ax.legend()
            ax.grid()
            fig.savefig(f'{metric}.png')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    small_size, medium_size, bigger_size = 10, 12, 16
    plt.rc('font', size=small_size)
    plt.rc('axes', titlesize=small_size, labelsize=medium_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=medium_size)
    plt.rc('figure', titlesize=bigger_size)

    n_nodes = 4
    n_rounds = 20
    n_estimators_per_node_per_round = 5
    m_estimators_to_select_per_round = 10

    datasets = ['a9a', 'cifar_binary', 'creditcard', 'gisette', 'hepmass_150K', 'ijcnn', 'skin_segmentation']
    split_methods = ['Random', 'KMeans']
    aggregation_methods = ['Random', 'KMeans', 'Center']

    for dataset in datasets:
        # train usual algorithm
        workload = FeatureImportanceDistanceWorkload(dataset, n_nodes, split='Random')
        workload.load_and_split_dataset()
        rf_all = RandomForestClassifier(
            # n_estimators=n_estimators_per_node_per_round * n_rounds,
            n_estimators=m_estimators_to_select_per_round * n_rounds,
            n_jobs=cpu_count(),
            random_state=np.random.randint(2 ** 31 - 1)
        )
        x_all, y_all = workload.all_data
        x_test, y_test = workload.test_data
        rf_all.fit(x_all, y_all)
        y_pred = rf_all.predict(x_test)
        usual_metrics = {
            'Balanced accuracy': balanced_accuracy_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred)
        }

        for split_method in split_methods:
            try:
                aggregation_metrics = {}
                for aggregation_method in aggregation_methods:
                    # train federation simulation
                    workload = FeatureImportanceDistanceWorkload(dataset, n_nodes, split=split_method)
                    workload.init(n_estimators_per_node_per_round, m_estimators_to_select_per_round, method=aggregation_method)
                    workload.run(n_rounds)
                    aggregation_metrics[aggregation_method] = workload.global_metrics

                # make graphics
                for metric in workload.local_metrics.keys():
                    # make graphics
                    fig, ax = plt.subplots()
                    fig.set_size_inches(9, 6)
                    fig.suptitle(f'"{dataset}" dataset - {split_method} data split - {metric} metric', fontsize=bigger_size)
                    ax.set_xticks([i for i in range(workload.n_rounds)])
                    for aggregation_method in aggregation_methods:
                        ax.plot([i for i in range(workload.n_rounds)], aggregation_metrics[aggregation_method][metric], label=aggregation_method + ' aggregation', marker='o')
                    ax.plot([0, workload.n_rounds - 1], [usual_metrics[metric], usual_metrics[metric]], label='Usual model', color='black')
                    ax.set(xlabel='Round', ylabel='Metric')
                    ax.legend(loc='best')
                    ax.grid()
                    fig.savefig(f'{dataset}-{split_method}-split-{metric}.png')
            except ValueError:
                pass
