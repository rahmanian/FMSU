#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:22:50 2019

@author: rahmanian
"""

import numpy as np
import numpy.matlib as mb
import fuzzy_utility as util
import time
import pandas as pd
import os.path, sys
from scipy.io import arff, loadmat
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn import metrics
import numba as nb

class featureClustering:
    def __init__(self, Debug=True):
        self.Debug = Debug
        self.name = ''
        self.numClass = 0
        self.data_set_names = {'ecoli':['data', 'ecoli.data'],
                               'thoracic':['arff', 'ThoraricSurgery.arff'],
                               'parkinsons':['data', 'parkinsons.data'],
                               'breast_cancer':['data', 'BreastCancer.data'],
                               'lung':['data', 'lung-cancer.data'],
                               'spambase':['data', 'spambase.data'],
                               'fertility':['data', 'fertility_Diagnosis.txt'],
                               'breast_tissue':['xls', 'BreastTissue.xls' ],
                               'image':['data', 'segmentation.data'],
                               'qsar':['data', 'QSAR.csv'],
                               'sonar':['data', 'sonar.all-data'],
                               'mfeat-fourier':['arff','image/dataset_14_mfeat-fourier.arff'],
                               'AR10P':['mat','image/warpAR10P.mat'],
                               'PIE10P':['mat','image/warpPIE10P.mat'],
                               'CLL_SUB_111':['mat','microarray/CLL_SUB_111.mat'],
                               'ALLAML':['mat','microarray/ALLAML.mat'],
                               'arrhythmia':['data','microarray/arrhythmia.data'],
                               'colon':['mat','microarray/colon.mat'],
                               'Embryonal_tumours':['arff','microarray/Embryonal_tumours.arff'],
                               'B-Cell1':['arff','microarray/B-Cell1.arff'],
                               'B-Cell2':['arff','microarray/B-Cell2.arff'],
                               'B-Cell3':['arff','microarray/B-Cell3.arff'],
                               'SMK_CAN_187':['mat','microarray/SMK_CAN_187.mat'],
                               'TOX_171':['mat','microarray/TOX_171.mat'],
                               'chess':['arff','text/chess.dat'],
                               'coil2000':['arff','text/coil2000.dat'],
                               'wine':['data', 'wine/wine2.data'],
                               'oh0.wc':['arff','text/oh0.wc.arff'],
                               'tr11.wc':['arff','text/tr11.wc.arff'],
                               'tr12.wc':['arff','text/tr12.wc.arff'],
                               'tr21.wc':['arff','text/tr21.wc.arff']}
    def load_data(self, data_set='ALLAML', base='ICCKE2020_dataset/'):
        try:
            [ds_type, ds_path] = self.data_set_names[data_set]   
            self.name = data_set
        except KeyError:
            print(f'Error: Dataset name <{data_set}> is not valid.')
            sys.exit(-1)
        else:
            if ds_type == 'arff':
                data = arff.loadarff(base+ds_path)
                df = pd.DataFrame(data[0])
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                tmp =  (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,:-1].to_numpy()
                if data_set == 'chess':
                    tmp = np.empty_like(self.X, dtype=np.int16)
                    tmp[self.X == b'f'] = 0
                    tmp[self.X == b'n'] = 0
                    tmp[self.X == b'l'] = 0
                    tmp[self.X == b't'] = 1
                    tmp[self.X == b'w'] = 1
                    tmp[self.X == b'g'] = 1
                    tmp[self.X == b'b'] = 2
                    self.X = np.copy(tmp)
            elif ds_type == 'mat':
                data = loadmat(base+ds_path)
                self.label = np.copy(data['Y'])
                self.X = np.copy(data['X'])
            elif ds_type == 'data':
                if data_set == 'parkinsons':
                    data = pd.read_csv(base+ds_path, header=0, index_col=0)
                elif data_set == 'breast_cancer':
                    data = pd.read_csv(base+ds_path, header=None, index_col=0)
                else:
                    data = pd.read_csv(base+ds_path, header=None)
                df = pd.DataFrame(data)
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                if data_set == 'parkinsons':
                    tmp = (df['status'].to_numpy())
                elif data_set in ['breast_cancer','lung', 'image']:
                    tmp = (df.iloc[:,0].to_numpy())
                else:
                    tmp = (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                if data_set == 'parkinsons':
                    self.X = df.drop('status', inplace=False, axis=1).to_numpy()
                elif data_set in ['lung','breast_cancer', 'image']:
                    self.X = df.iloc[:,1:].to_numpy()
                else:
                    self.X = df.iloc[:,:-1].to_numpy()
            elif ds_type == 'xls':
                data = pd.read_excel(base+ds_path, header=0, index_col=0, sheet_name='Data')
                df = pd.DataFrame(data)
                print(df.shape)
                tmp = (df.iloc[:,0].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,1:].to_numpy()
            else:
                print(f'Type of dataset <{ds_type}> is not valid.')
                return
            self.numFeatures = self.X.shape[1]
            if isinstance(self.label[0,0],bytes):
                try:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = int(self.label[i,0]) 
                except:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = self.label[i,0].decode('utf-8')
            self.name = data_set
            tmp = np.unique(self.label, return_index=False, return_inverse=False,
                          return_counts=False)
            self.numClass = len(tmp)
            self.uniqueLabel = tmp[:]
            self.FIS = util.FuzzyInfoSystem(self.X, self.label, f"FuzzyMatrix/FMR_{self.name}.npy")
            
    def symmetric_uncertainty(self, path, recalculate=False):
        if self.Debug:
            print(f'Calculate Symmetric Uncertainty for {self.name} dataset.')
        if  (not recalculate) and os.path.exists(path):
            self.SU = np.load(path)
        else:
            import time
            start = time.time()
            self.SU = np.zeros((self.numFeatures, self.numFeatures))
            for i in range(self.numFeatures):
                if self.Debug and i%300==0:
                    print(f'\t\tFeatures #{i} to {i+300} vs others.')
                for j in range(i+1, self.numFeatures):
                    self.SU[i,j] = self.FIS.fuzzy_symmetrical_uncertainty(self.FIS.MR[i], self.FIS.MR[j])
                    self.SU[j,i] = self.SU[i,j]
            print(f"SU time = {time.time()-start} seconds")
            np.save(path, self.SU)
    
    def KNN(self, path, recalculate=False):
        if self.Debug:
            print(f'Calculate kNN for {self.name} dataset.')
            start = time.time()
        if  (not recalculate) and os.path.exists(path):
            self.kNN = np.load(path)
        else:
            self.k = int(np.sqrt(self.numFeatures))
            self.FIS.matrix = np.copy(self.SU)
            self.kNN = self.FIS.kNN(self.FIS.MR, k=self.k+1)
            np.save(path, self.kNN)
        if self.Debug:
            end = time.time()
            print(f'\tTime for kNN on {self.name} is {end - start} seconds')
    
    def Density_kNN(self, D_kNN_filename=None, base_path='D_kNN', recalculate=False):
        print('Compute D_kNN...')
        self.k = int(np.sqrt(self.numFeatures))
        if (not recalculate) and os.path.exists(D_kNN_filename):
            self.D_kNN = np.load(D_kNN_filename)
        else:
            self.D_kNN = np.zeros(self.numFeatures, dtype=np.float)
            for f in range(self.numFeatures):
                #print(f'#{f} from {self.numFeatures}')
                sum_of_SU = 0.0
                for neigbor in self.kNN[f,1:]:
                    sum_of_SU += self.SU[f,neigbor]
                self.D_kNN[f] = sum_of_SU/(self.k)
            name = "D_kNN_"+self.name+".npy" if D_kNN_filename == None else D_kNN_filename 
            np.save(f'{name}', self.D_kNN)
        print('D_kNN computed.') 
        
    def initial_centers(self):
        self.m = 0
        self.maxSU = -10e36
        sorted_D_kNN = np.argsort(-self.D_kNN)   
        
        Fs_ind = np.copy(sorted_D_kNN)
        Fs_ind = Fs_ind.astype(np.int16)
        self.Fc = []
        self.Fc.append(self.FIS.MR[Fs_ind[0]])
        self.Fc_ind = np.array([Fs_ind[0]], dtype=np.int16)
        Fs_ind = np.delete(Fs_ind, np.where(Fs_ind == self.Fc_ind))
        self.m += 1
        
        for fs in Fs_ind:
            for fc in self.Fc_ind:
                if (fc in self.kNN[fs]) or (fs in self.kNN[fc]):
                    break
            else:
                self.Fc_ind = np.append(self.Fc_ind, [fs])
                self.Fc.append(self.FIS.MR[fs])
                self.m += 1
                self.maxSU = max(self.maxSU, self.SU[fs,fc])
        self.Fc = np.array(self.Fc)
        
    def main_loop(self, path, max_feature=100, MAX_ITER=20, recalculate=False):
        self.Fc_ind = self.Fc_ind[:max_feature]
        #Fc_ind = Fs_ind[:]
        self.Fc = self.Fc[:max_feature]
        if max_feature>self.Fc.shape[0]:
            max_feature = self.Fc.shape[0]
        
        print('main loop of my algorithm...')
        if  (not recalculate) and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            clusters = np.copy(data['clusters'])
            self.Fc = np.copy(data['fc'])
        else:
            changed = True
            
            itr = 1
            while changed and itr < MAX_ITER:
                print(max_feature, self.Fc.shape[0])
                #assert(max_feature == self.Fc.shape[0])
                print(f'Assign samples ...#{itr}')
                Fs_ind = np.arange(self.X.shape[1]-1)
                finished = False
                while not finished:
            #        if m > numFeatures:
            #            break
                    clusters = [[None] for i in range(self.Fc.shape[0])]
                    for k, fs in enumerate(Fs_ind):
                        if k%100==0:
                            print(f'Iter#{itr}\t Feature#{k}')
                        SU_centers = np.zeros(self.Fc.shape[0], dtype=np.float)
                        for i, fc in enumerate(self.Fc):
                            tmp = self.FIS.fuzzy_symmetrical_uncertainty(self.FIS.MR[fs], fc)
                            # tmp = FS.mutual_information(fs, fc)
                            SU_centers[i] = tmp
                        j = np.argmax(SU_centers)
                        clusters[j].append(fs)
                    else:
                        finished = True
                print('Samples Assigned.')
                for i in range(self.Fc.shape[0]):
                    clusters[i] = clusters[i][1:]
                print(f'Update center of clusters...#{itr}')
    #            new_Fc_ind = np.copy(self.Fc_ind)
                new_Fc = np.copy(self.Fc)
                change_Fc = False
                zero_num = []
                for i in range(self.Fc.shape[0]):
                    #print(f'Iter#{itr}\t Cluster#{i}')
                    fc = np.copy(self.Fc[i]) 
                    if len(clusters[i]) == 0:
                        zero_num.append(i)
                        new_Fc[i,:,:] = fc
                        continue
                    tmp_fc = self.FIS.indiscernibility_feature_more(self.FIS.MR[clusters[i]])
                    new_Fc[i,:,:] = tmp_fc
                    #exit(0)
                    if np.any(fc != tmp_fc):
                        change_Fc = True
                else:
                    print(f"Zero members cluster: {zero_num}")
                    print(f"Shape of FC before delete is {new_Fc.shape}")
                    if len(zero_num) != 0:
                        new_Fc = np.delete(new_Fc,zero_num,axis=0)
                print(f"Shape of FC after delete is {new_Fc.shape}")
                if not change_Fc:
                    changed = False
                self.Fc = new_Fc.copy()
                print('Center of clusters selected.')
                itr += 1
            np.savez(path, clusters=clusters, fc=self.Fc)
        print('main loop of my algorithm finished.')
        
        self.selected_ = []
        # Fs = np.copy(self.X)
        for j, fc in enumerate(self.Fc):
            mi = np.zeros(len(clusters[j]))
            for k,i in enumerate(clusters[j]):
                mi[k] = self.FIS.fuzzy_mutual_information(fc, self.FIS.MR[i])
            self.selected_.append(clusters[j][np.argmax(mi)])
       
    def _learner(self, full=False):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.numClass, init='k-means++', random_state=0)
        if not full:
            X = self.X[:,self.selected_]
        else:
            X = self.X[:,:]
        kmeans.fit(X)
        y_hat = np.copy(kmeans.labels_)
        y_hat = y_hat.reshape(-1,1)
        # print(y_hat.shape, self.label.shape, X.shape)
        return y_hat   
    
    def compute_scores(self):
        from sklearn.metrics import f1_score, adjusted_mutual_info_score
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        y_hat_full = self._learner(full = True)
        y_hat_partial = self._learner(full = False)
        y_hat_full = y_hat_full.reshape(y_hat_full.shape[0],)
        y_hat_partial = y_hat_partial.reshape(y_hat_full.shape[0],)
        
        y_uniq = np.unique(self.label)
        y = np.zeros(self.label.shape[0])
        for i, _y in enumerate(y_uniq):
            tmp = self.label.T == _y
            # print(tmp[0], _y)
            y[tmp[0]] = i
                
        f_score_full = f1_score(y, y_hat_full, average = 'micro')
        f_score_part = f1_score(y, y_hat_partial,average = 'micro')
        
        ami_full = adjusted_mutual_info_score(y, y_hat_full)
        ami_part = adjusted_mutual_info_score(y, y_hat_partial)
        
        ars_full = adjusted_rand_score(y, y_hat_full)
        ars_part = adjusted_rand_score(y, y_hat_partial)
        
        silh_full = silhouette_score(self.X, y_hat_full)
        silh_part = silhouette_score(self.X[:,self.selected_], y_hat_partial)
                
        
        return f_score_full, f_score_part, ami_full, ami_part, ars_full, ars_part, silh_full, silh_part
    
    def computing_accuracy(self,path, recalculate=False):
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        y = self.label.reshape(1,-1)[0]
        if isinstance(y[0], int):
            y = y.astype(np.int)            
        knn = KNeighborsClassifier(n_neighbors=3)
        cross = cross_val_score(knn, self.X, y, cv=5, scoring='accuracy')
        knn_all = cross.mean()
        print(f'Accuracy(knn) with all features = {knn_all:0.4f}.')
        
        svm_clf = svm.SVC()
        cross = cross_val_score(svm_clf, self.X, y, cv=5, scoring='accuracy')
        svm_all = cross.mean()
        print(f'Accuracy(svm) with all features = {svm_all:0.4f}.')
        knn_selected = 0.0
        svm_selected = 0.0
        if  (not recalculate) and os.path.exists(path):
            acc = np.load(path)
            knn_all = np.copy(acc['knn_all'])
            knn_selected = np.copy(acc['knn'])
            svm_all = np.copy(acc['svm_all'])
            svm_selected = np.copy(acc['svm'])
        else:
                       
            num_test = 100 if self.numFeatures>100 else self.numFeatures//2
            
            for selected_size in range(len(self.selected_),len(self.selected_)+1):
                #selected_features, index = util.average_redundancy(clusters, selected_size, SU_Mat)
                ss = self.X[:,self.selected_]
                tmp = cross_val_score(knn, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                knn_selected = tmp.mean()
                print(f'Accuracy(knn) with {selected_size} select features = {knn_selected:0.4f}.')
                
                tmp = cross_val_score(svm_clf, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                svm_selected = tmp.mean()
                print(f'Accuracy(svm) with {selected_size} select features = {svm_selected:0.4f}.')

            #    print(f'Accuracy(lsvm) on selected(FSFC) features: {scores_selected_svm[selected_size]}')
            np.savez(path, knn_all=knn_all, knn=knn_selected, svm_all=svm_all, svm=svm_selected)
        return knn_all, knn_selected, svm_all, svm_selected
