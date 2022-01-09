#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:26:02 2020

@author: rahmanian
"""
import numpy as np
import time
from FMSU import featureClustering
import sys

datasets = featureClustering.data_set_names.keys()

if __name__ == "__main__":
    base = 'dataset/'
    for dataset_name in datasets:
        fis = featureClustering()
        print('Loading data and create Fuzzy systems...')
        fis.load_data(dataset_name, base=base)
        print('Data loaded and Fuzzy systems created.')
        fis.symmetric_uncertainty(f'SU/fuzzy_SU_{dataset_name}.npy',recalculate=True)
        print(f'\t SU shape {fis.SU.shape}')
        fis.KNN(f'KNN/fuzzy_KNN_{dataset_name}.npy',recalculate=True)
        print(f'\t KNN shape {fis.kNN.shape}')
        fis.Density_kNN(f"DKNN/fuzzy_DKNN_{dataset_name}.npy", recalculate=True)
        print(f'\t DKNN shape {fis.D_kNN.shape}')       
        fis.initial_centers()
        print(f'\t Fc shape {fis.Fc.shape}')
        start = time.time()
        fis.main_loop(f'Clusters/fuzzy_clusters_{dataset_name}.npz', recalculate=True)
        main_loop_time = time.time() - start
        print(fis.selected_)
        print(f"Main loop finished in {main_loop_time:0.6} seconds.")
        fs_f, fs_p, ami_f, ami_p, ars_f, ars_p, sil_f, sil_p = fis.compute_scores()
        result = f'\t f_score_full={fs_f:0.4f}, f_score_part={fs_p:0.4f}\n'\
                 f'\t NMI_full    ={ami_f:0.4f}, NMI_part    ={ami_p:0.4f}\n'\
                 f'\t ARI_full    ={ars_f:0.4f}, ARI_part    ={ars_p:0.4f}\n'\
                 f'\t SILH_full   ={sil_f:0.4f}, SILH_part   ={sil_p:0.4f}\n'\
                 f'features: {fis.selected_}\n'\
                 f'#feature   ={len(fis.selected_)}'
        print(result)
        with open(f'Accuracy/accuracy_{dataset_name}.txt','w') as file:
            file.write(result)
        # fis.selected_ = np.array([241, 242, 10, 261, 251, 29, 63, 87, 147, 90, 161, 53, 42, 221, 236, 266, 246, 78, 45])
        knn_full, knn, svm_full, svm = fis.computing_accuracy(f"Accuracy/fuzzy_accuracy_{dataset_name}.npz", recalculate=True)
        with open(f'Accuracy/accuracy_{dataset_name}.txt','a') as file:
            file.write('\n')
            file.write(f'knn_full \t= {knn_full}\n'
                       f'knn_selected = {knn}\n'
                       f'svm_full \t= {svm_full}\n'
                       f'svm_selected = {svm}\n'
                       f'time = {main_loop_time} seconds')