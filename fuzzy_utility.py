#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:07:51 2019

@author: rahmanian
"""

import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

class FuzzyInfoSystem:
    def __init__(self, data, label, fileName="FRelationMatrix.npy"):
        self.X = np.copy(data)
        self.n_samples, self.n_features = self.X.shape
        self.C = np.copy(label)
        self.X = np.c_[self.X,self.C]
        self.fileName = fileName
        self.fuzzy_relation_matrix()
        
    def similarity(self, a, b):
        return np.exp(-1*np.abs(a-b))
    
    def similarity_class(self, a, b):
        return 1 if a==b else 0
    
    def fuzzy_relation_matrix_feature(self,f, class_=False):
        MR = np.zeros((self.n_samples, self.n_samples))
        similarity = self.similarity
        if class_:
            similarity = self.similarity_class
        for i in range(self.n_samples):
            for j in range(i, self.n_samples):
                MR[i,j] = similarity(f[i], f[j])
                MR[j,i] = MR[i,j]
        return MR
    
          
    def fuzzy_relation_matrix(self):
        if os.path.exists(self.fileName):
            self.MR = np.load(self.fileName)
        else:
            self.MR = np.zeros((self.n_features+1, self.n_samples, self.n_samples))
            for i in range(self.n_features):
                self.MR[i,:,:] = np.copy(self.fuzzy_relation_matrix_feature(self.X[:,i]))
            self.MR[self.n_features,:,:] = np.copy(self.fuzzy_relation_matrix_feature(self.X[:,-1],class_=True))
            np.save(self.fileName,self.MR)
            
    def relation_cardinality(self, sample, M_feature):
        # sample, feature = int(sample), int(feature)
        sample = int(sample)
        card = 0
        s = M_feature.shape[1]
        for i in range(s):
            # print(f'in realation_card {sample}, {feature}')
            card += M_feature[sample,i]
        return card
    
    def fuzzy_entropy(self, M_f):
        # f = int(f)
        h = 0.0
        for i in range(self.n_samples):
            h -= np.log2(self.relation_cardinality(i,M_f)/self.n_samples)
        h = h/self.n_samples
        return h
    
    def indiscernibility_feature(self, M_f1, M_f2):
        # f1, f2 = int(f1), int(f2)
        ind = np.minimum(M_f1, M_f2)
        return ind

    def indiscernibility_feature_more2(self, M_f_list):
        min_ = M_f_list[0,:,:]
        for i in range(1,M_f_list.shape[0]):
            min_ = np.minimum(M_f_list[i,:,:], min_)
        return min_
    
    def indiscernibility_feature_more(self, M_f_list):
        min_ = np.minimum.reduce(M_f_list)
        return min_
    
    def ind_cardinality(self, sample, M_f1, M_f2):
        card = 0
        ind = self.indiscernibility_feature(M_f1,M_f2)
        card = self.relation_cardinality(sample, ind)
        # for i in range(ind.shape[1]):
        #     card += ind[sample,i]
        return card
    
    def ind_cardinality_more(self, sample, M_f_list):
        card = 0.0
        ind = self.indiscernibility_feature_more(M_f_list)
        card = self.relation_cardinality(sample, ind)
        # for i in range(ind.shape[1]):
        #     card += ind[sample,i]
        return card
            
        
    def joint_fuzzy_entropy(self, M_f1, M_f2):
        h = 0.0
        for i in range(self.n_samples):
            h -= np.log2(self.ind_cardinality(i,M_f1,M_f2)/self.n_samples) 
        h = h / self.n_samples
        return h
    
    def joint_fuzzy_entropy_more(self, M_f_list):
        h = 0.0
        for i in range(self.n_samples):
            h -= np.log2(self.ind_cardinality_more(i,M_f_list)/self.n_samples) 
        h = h / self.n_samples
        return h
    
    def fuzzy_mutual_information(self, M_x, M_y):
        """
        Parameters
        ----------
        x : a numpy array
            DESCRIPTION.
        y : a numpy array
            DESCRIPTION.

        Returns
        -------
        A real number
            MI(x,y) = H(x) + H(y) - H(x,y).

        """
        Hx = self.fuzzy_entropy(M_x)
        Hy = self.fuzzy_entropy(M_y)
        Hxy = self.joint_fuzzy_entropy(M_x,M_y)
        return Hx + Hy - Hxy
    
    def conditional_fuzzy_entropy(self, M_x, M_y):
        Hy = self.fuzzy_entropy(M_y)
        Hxy = self.fuzzy_joint_entropy(M_x,M_y)
        return Hxy - Hy
    
    def canditional_fuzzy_mutual_information(self, M_x, M_y, M_z):
        Hxz = self.joint_fuzzy_entropy(M_x, M_z)
        Hyz = self.joint_fuzzy_entropy(M_y, M_z)
        Hxyz = self.joint_fuzzy_entropy_more([M_x, M_y, M_z])
        Hz = self.fuzzy_entropy(M_z)
        return Hxz + Hyz - Hxyz - Hz
    
    def joint_fuzzy_mutual_information(self, M_x, M_y, M_z):
        Ixz = self.fuzzy_mutual_information(M_x, M_z)
        Iyz_x = self.conditional_mutual_information(M_y, M_z, M_x)
        return Ixz + Iyz_x
    
    
    def fuzzy_symmetrical_uncertainty(self, M_X1, M_X2):
        hx1 = self.fuzzy_entropy(M_X1)
        hx2 = self.fuzzy_entropy(M_X2)
        mi = self.fuzzy_mutual_information(M_X1, M_X2)
        return 2*mi/(hx1+hx2)
    
    def fuzzy_multivariate_symmetrical_uncertainty(self, M_f_list):
        n = M_f_list.shape[0]
        sum_ = 0.0
        for M_f in M_f_list:
            sum_ += self.fuzzy_entropy(M_f)
        hx1_n = self.joint_fuzzy_entropy_more(M_f_list)
        msu = (n/(n-1))*(1-hx1_n/sum_)
        return msu
    
    def SU_similarity(self, x, y):
        su = 0.0
        x = int(x[0])
        y = int(y[0])
            
        su = self.matrix[x,y]
        t = su if su > 0 else 10e-15
        return 1.0/t  
    
    def kNN(self, M_f_list, k=3):
        if not hasattr(self, "matrix"):
            raise Exception("Objact must be has <matrix> attribute.")
        n,m = self.n_samples, self.n_features
        indx = np.arange(m,dtype=int)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=self.SU_similarity)
        nbrs.fit(indx.reshape(-1,1))
        distances, indices = nbrs.kneighbors(np.array(indx).reshape(-1,1))
        return indices     