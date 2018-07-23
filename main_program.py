#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:42:34 2018
@author: nikoescanilla
Subject: Run 10-fold CV on Breast Cancer dataset
"""

# install packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp

from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm.libsvm import predict_proba
from sklearn import model_selection, preprocessing
from sklearn.metrics import (precision_score, recall_score, f1_score, 
precision_recall_fscore_support, roc_curve, auc, roc_auc_score) 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

"""
Functions:
"""

# RFEST Algorithm (CV)
# In: data, labels, folds
# Out: AUC (from CV)
def RFEST(skf, df_feature_data_complete, df_label_data, df_feature_names_complete):
    # create list of scores to keep track of
    f1_df = pd.DataFrame(columns=['F1 Score'])
    f1_df = f1_df.fillna(0)
    f1score_list = []
    auc_df = pd.DataFrame(columns=['AUC'])
    auc_df = auc_df.fillna(0)
    auc_list = []
    tpr_rfest = []
    mean_fpr = np.linspace(0, 1, 100)

    fig = plt.figure(figsize=(10,8))
    
    # initialize number of features to remove (by percent)
    percent = 0.1

    count = 0
    
    # for each fold
    for train_index, test_index in skf.split(df_feature_data_complete, df_label_data):        
        print('Currently working on fold', count + 1)
        
        # initialize best and max AUCs
        best_auc = 0
        max_auc = 0
    
        # inititalize SVM model
        svm_clf = svm.SVC(kernel = 'rbf', probability = True, random_state = 1)
    
        # keep copy of both train and tune
        train_index_whole = train_index
        
        # randomly select 20% for tune set
        np.random.seed(1)
        sample_size = int(np.floor(0.2*len(train_index)))
        tune_index = np.random.choice(train_index, size = sample_size, replace = False)
        train_index = np.setdiff1d(train_index, tune_index)
        
        df_feature_data_for_final_model = df_feature_data_complete
        df_feature_data = df_feature_data_complete
        df_feature_names = df_feature_names_complete
        
        #print('Number of features we start out with for current fold: ', np.shape(df_feature_data)[1])
        
        while True:
            # train and test model
            svm_clf.fit(df_feature_data[train_index],df_label_data[train_index])
            y_pred_prob = svm_clf.predict_proba(df_feature_data[tune_index])[:,1]
            y_pred_bin = svm_clf.predict(df_feature_data[tune_index])
            
            # compute different accuracy measurements
            f1 = f1_score(df_label_data[tune_index], y_pred_bin, pos_label=1, average='binary')
            fpr, tpr, thresholds = roc_curve(df_label_data[tune_index], y_pred_prob, pos_label=1)
            curr_auc = auc(fpr, tpr)
            print('Current AUC: ', curr_auc)
        
            if curr_auc <= (0.95*max_auc):
                # build final model using only remaining features
                svm_clf.fit(df_feature_data_for_final_model[train_index_whole],df_label_data[train_index_whole])
                y_pred_prob = svm_clf.predict_proba(df_feature_data_for_final_model[test_index])[:,1]
                y_pred_bin = svm_clf.predict(df_feature_data_for_final_model[test_index])
                
                # compute different accuracy measurements
                f1 = f1_score(df_label_data[test_index], y_pred_bin, pos_label=1, average='binary')
                f1score_list.append(f1)
                f1_df.loc[count] = f1

                fpr, tpr, thresholds = roc_curve(df_label_data[test_index], y_pred_prob, pos_label=1)
                tpr_rfest.append(interp(mean_fpr, fpr, tpr)) 
                tpr_rfest[-1][0] = 0.0
    
                final_auc = auc(fpr, tpr)
                auc_list.append(final_auc)
                auc_df.loc[count] = final_auc
                
                plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                         label='ROC fold %d (AUC = %0.2f)' % (count+1, final_auc))
                print('AUC on test set for fold ', count + 1, ' is ', final_auc)
                count = count + 1
                break
            else:
                # record current best AUC 
                best_auc = curr_auc
                if best_auc >= max_auc:
                    max_auc = best_auc
            
            # compute importance of each feature
            for idx in df_feature_names['feat_index']:
                original_vec = df_feature_data[tune_index, idx]
                flipped_vec = 1 - original_vec
                
                df_feature_data[tune_index, idx] = flipped_vec
                
                y_pred_prob = svm_clf.predict_proba(df_feature_data[tune_index])[:,1]
                y_pred_bin = svm_clf.predict(df_feature_data[tune_index])
                
                f1 = f1_score(df_label_data[tune_index], y_pred_bin, pos_label=1, average='binary')
                fpr, tpr, thresholds = roc_curve(df_label_data[tune_index], y_pred_prob, pos_label=1)
                flip_auc = auc(fpr, tpr)
                
                # note: the more positive, the more relevant
                diff_auc = curr_auc - flip_auc
                
                # update df_feature_names dataframe
                df_feature_names.loc[idx, 'feat_importance'] = diff_auc
                
                # flip column values back to original values
                df_feature_data[tune_index, idx] = original_vec
                
            # find smallest values (10% of total features thus far)
            idx_to_delete = np.array(df_feature_names.nsmallest(int(np.ceil(percent*np.shape(df_feature_data)[1])), 
                                                          'feat_importance')['feat_index'])
            
            # save a copy of feature data in case the model breaks
            df_feature_data_for_final_model = df_feature_data
            
            # remove lowest ranked features from df_feature_data and df_feature_names
            df_feature_data = np.delete(df_feature_data, idx_to_delete, axis = 1)
            df_feature_names = df_feature_names.drop(idx_to_delete)
            df_feature_names['feat_index'] = range(np.shape(df_feature_data)[1])
            df_feature_names.index = range(np.shape(df_feature_data)[1])
            
            # print number of features remaining
            #print('Number of features left: ', np.shape(df_feature_data)[1])
            
    print("plotting figure...")    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck', alpha=.8)

    mean_tpr = np.mean(tpr_rfest, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    plt.plot(mean_fpr, mean_tpr, color='r',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tpr_rfest, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for RFEST SVM (built on subset of features returned by cross-validated RFEST)')
    plt.legend(loc="lower right")
    plt.show()

    fig.savefig('results/10PercentRemoval/rfest_roc_curve.png')

    print("\nAverage F1 Score: "+ str(sum(f1score_list)/n_splits)+
          "\nAverage AUC: "+ str(sum(auc_list)/n_splits))

    # save auc and f1
    auc_df.to_csv("results/10PercentRemoval/rfest_rbf_svm_ten_fold_auc.csv", index = False)
    f1_df.to_csv("results/10PercentRemoval/rfest_rbf_svm_ten_fold_f1_score.csv", index = False)
    return 
    
# RFEST 
# In: data, labels
# Out: List of features
def RFEST_features(df_feature_data_complete, df_label_data, df_feature_names_complete):
    # initialize best and max AUCs
    best_auc = 0
    max_auc = 0
    
    # inititalize SVM model
    svm_clf = svm.SVC(kernel = 'rbf', probability = True, random_state = 1)
        
    # randomly select 20% for tune set
    train_index = range(np.shape(df_feature_data_complete)[0])
    np.random.seed(1)
    sample_size = int(np.floor(0.2*len(train_index)))
    tune_index = np.random.choice(train_index, size = sample_size, replace = False)
    train_index = np.setdiff1d(train_index, tune_index)
    
    df_feature_data = df_feature_data_complete
    df_feature_names = df_feature_names_complete
    df_feature_names_final = df_feature_names_complete    
      
    # initialize number of features to remove (by percent)
    percent = 0.1
        
    while True:
        # train and test model
        svm_clf.fit(df_feature_data[train_index],df_label_data[train_index])
        y_pred_prob = svm_clf.predict_proba(df_feature_data[tune_index])[:,1]
            
        # compute different accuracy measurements
        fpr, tpr, thresholds = roc_curve(df_label_data[tune_index], y_pred_prob, pos_label=1)
        curr_auc = auc(fpr, tpr)
        print('Current AUC: ', curr_auc)
        
        if curr_auc <= (0.95*max_auc):
            # return final list of features
            df_feature_names_final.to_csv("results/10PercentRemoval/fs_features_8020_split.csv", index = False)
            break
        else:
            # record current best AUC 
            best_auc = curr_auc
            if best_auc >= max_auc:
                max_auc = best_auc
            
        # compute importance of each feature
        for idx in df_feature_names['feat_index']:
            original_vec = df_feature_data[tune_index, idx]
            flipped_vec = 1 - original_vec
                
            df_feature_data[tune_index, idx] = flipped_vec
                
            y_pred_prob = svm_clf.predict_proba(df_feature_data[tune_index])[:,1]
                
            fpr, tpr, thresholds = roc_curve(df_label_data[tune_index], y_pred_prob, pos_label=1)
            flip_auc = auc(fpr, tpr)
                
            # note: the more positive, the more relevant
            diff_auc = curr_auc - flip_auc
                
            # update df_feature_names dataframe
            df_feature_names.loc[idx, 'feat_importance'] = diff_auc
                
            # flip column values back to original values
            df_feature_data[tune_index, idx] = original_vec
                
        # find smallest values (10% of total features thus far)
        idx_to_delete = np.array(df_feature_names.nsmallest(int(np.ceil(percent*np.shape(df_feature_data)[1])), 
                                                        'feat_importance')['feat_index'])
            
        # save a copy of features in case the model breaks
        # df_feature_data_for_final_model = df_feature_data
        df_feature_names_final = df_feature_names
            
        # remove lowest ranked features from train data, test data, and df_feature_names
        df_feature_data = np.delete(df_feature_data, idx_to_delete, axis = 1)
        df_feature_names = df_feature_names.drop(idx_to_delete)
        df_feature_names['feat_index'] = range(np.shape(df_feature_data)[1])
        df_feature_names.index = range(np.shape(df_feature_data)[1])
            
        # print number of features remaining
        print('Number of features left: ', np.shape(df_feature_data)[1])

    # save auc and f1
    return 


"""
Main Program
"""

# read in data
df = pd.read_csv('data/cleanData/final_data.csv')
df_feature_data = df.drop(['class_label'], axis = 1)
df_feature_data = df_feature_data.values
df_label_data = df['class_label'].values

feature_names = list(df)[0:len(list(df))-1]
df_feature_names = pd.DataFrame(
        {'feat_index': range(len(list(df))-1),
         'feat_name': feature_names,
         'feat_importance': [0]*(len(list(df))-1)})

    
# perform stratified sampling for CV
n_splits = 10
skf = StratifiedKFold(n_splits,random_state=1)
skf.get_n_splits(df_feature_data,df_label_data)

# compute CV for RFEST
print('Now computing AUC (CV) for RFEST')
RFEST(skf, df_feature_data, df_label_data, df_feature_names)

# return list of relevant features
print('Now determing list of relevant features')
RFEST_features(df_feature_data, df_label_data, df_feature_names)






