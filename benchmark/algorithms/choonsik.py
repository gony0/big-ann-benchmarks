from __future__ import absolute_import

import os
import multiprocessing

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS, download_accelerated

import choonsik

class Choonsik(BaseANN):
    def __init__(self, metric, index_params):
        self._ip = index_params
        self._metric = metric
        if 'n_threads' not in self._ip:
            self._ip['n_threads'] = multiprocessing.cpu_count()

    def get_index_dir(self, dataset):
        params = f'm{self._ip["hnsw_m"]}_ef{self._ip["hnsw_ef"]}_pq{self._ip["opq_m"]}_pqk{self._ip["opq_pk"]}'
        if 'model_suffix' in self._ip:
            params += f'_{self._ip["model_suffix"]}'
        return os.path.join("data", f'choonsik_{dataset}_{params}')

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        self._ip['dim'] = ds.d
        index_dir = self.get_index_dir(dataset)
        os.makedirs(index_dir, exist_ok=True)
        index_prefix = os.path.join(index_dir, 'model')
        os.remove(index_prefix) if os.path.exists(index_prefix) else None

        self._index = choonsik.BigAnnIndex(self._metric, self._ip)
        print(f'Creating Index. Index directory: {index_dir}')
        input_fn = ds.get_dataset_fn()
        print(f'Input: {input_fn}')
        self._index.fit(input_fn, index_prefix)
        os.rename(index_prefix, index_prefix + '.choonsik') if os.path.exists(index_prefix) else None

    def load_index(self, dataset):
        index_dir = self.get_index_dir(dataset)
        index = os.path.join(index_dir, 'model.choonsik')
        if not os.path.exists(index):
            return False
        print(f'Loding index from file. Index file: {index}')
        self._index = choonsik.load_index(self._metric, index)
        return True

    def set_query_arguments(self, query_params):
        print(f'query_params: {query_params}')
        self._qp = query_params
        self._index.set_search_opt(self._qp)

    def query(self, X, k):
        self.res = self._index.knn(X, k)

    def range_query(self, X, radius):
        '''
        Carry out a batch query for range search with
        radius.
        '''
        raise NotImplemented

    def get_results(self):
        return self.res

    def get_range_results(self):
        '''
        Helper method to convert query results of range search.
        If there are nq queries, returns a triple lims, I, D.
        lims is a (nq) array, such that
            I[lims[q]:lims[q + 1]] in int
        are the indiices of the indices of the range results of query q, and
            D[lims[q]:lims[q + 1]] in float
        are the distances.
        '''
        raise NotImplemented

    def __str__(self):
        return str(self._qp)[1:-1]
