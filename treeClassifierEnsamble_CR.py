#!/usr/bin/env python
# coding: utf-8



from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import clone
import imblearn  # imblanaced-learn >0.61
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import numpy as np
import pandas as pd
import os




def readFile(fileName):
    '''Returns list of lines in file'''
    with open(fileName, 'r') as inputFile:
        lines = inputFile.readlines()
        return lines




def dropcol_importances(rf, X_train, y_train):
    '''Return a list of tuples (feature_name, importance, index) reverse sorted by importance usuing drop column method'''
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    return sorted(zip(X_train.columns, imp, range(len(X_train.columns))), reverse=True, key=lambda kv: kv[1])


def main():
    # Files
    gt_file = './groundtruth_dataset.pkl'
    pred_set = './prediction_set_with_class.pkl'
    root_data_balance = './ML-CamReady/OOBpred/'
    root_accuracy = './ML-CamReady/MA/'
    root_prediction = './ML-CamReady/'
    # build path
    if not os.path.exists(root_data_balance):
        os.makedirs(root_data_balance)
    if not os.path.exists(root_accuracy):
        os.makedirs(root_accuracy)
    if not os.path.exists(root_prediction):
        os.makedirs(root_prediction)
    # IPv
    ipv = 'v4'

    # Creating pandas dataframe and converting to numerical variables.

    # pd.read_pickle(df_file)
    data = pd.read_pickle(gt_file)
    data_predict_orig = pd.read_pickle(pred_set)
    print ('Ground Truth data')
    print (data.shape)
    print ('Prediction set')
    print (data_predict_orig.shape)
    cols = data.columns

    print(data_predict_orig.columns)
    pre_cols = data_predict_orig.columns
    print(len(pre_cols))
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    data_predict_orig[pre_cols] = data_predict_orig[pre_cols].apply(pd.to_numeric, errors='coerce')
    data_predict_orig = data_predict_orig.drop('HardVotePred', axis=1)
    # print data_orig.dtypes

    # Feature selection
    features = list(data.columns)
    print ("Features: {:}".format(len(features)))
    print (features)
    # Getting ASN class
    print (data.head())
    print ('Ground truth shape :')
    print (data['class'].value_counts())
    X_train, y_train = data.drop('class', axis=1), data['class']

    # Sampling method for balancing
    rus = RandomUnderSampler()
    ros = RandomOverSampler()
    smote = SMOTE(sampling_strategy='minority')  # resample only the minority class;
    smt = SMOTETomek(sampling_strategy='auto')
    tl = TomekLinks(sampling_strategy='majority')
    cc_models = [ClusterCentroids(sampling_strategy={0: x}) for x in range(15, 30,
                                                                           1)]  # 十五个簇类下采样方法 cc_models size(15) Undersample by generating centroids based on clustering methods.
    SMs = [rus, ros, smote, smt, tl] + cc_models
    without_asn = [2, 3]  # 用于smote 和smt采样模型，丢掉ASN参数

    model_accuracy = []
    oob_decisions = []
    estimators_list = []
    forest_count = 0
    forest_size = 500
    models_reps = [0, 10, 10, 10, 4] + [0] * len(cc_models)  # 模型几轮

    for i, sm in enumerate(SMs):
        rep = models_reps[i]

        if i in without_asn:
            X_train, y_train = data.drop(['ASN', 'class'], axis=1), data['class']
        else:
            X_train, y_train = data.drop('class', axis=1), data['class']

        for r in range(rep):
            X_sm, y_sm = sm.fit_sample(X_train, y_train)  # resample

            # Balanced dataset
            # 			print(len(X_train.columns))
            # 			print(X_train.columns)
            # 			print(X_sm)
            data_balanced = pd.DataFrame(X_sm, columns=X_train.columns)
            # 			data_balanced = pd.DataFrame({X_train.columns[k] : X_sm[:,k].values for k in range(len(X_train.columns))})
            data_balanced.loc[:, 'class'] = y_sm.tolist()
            if r == 0:
                print ('Original dataset shape :')
                print (data_balanced['class'].value_counts())
            # print data_balanced.head()
            if i in without_asn:  # 这部分已经有了，是否删除
                X, y = data_balanced.drop('class', axis=1), data_balanced['class']
            else:
                X, y = data_balanced.drop(['ASN', 'class'], axis=1), data_balanced['class']

            # Train a forest and get accuracy
            forest = ExtraTreesClassifier(n_estimators=forest_size, bootstrap=True, oob_score=True)  # 极端树的划分点更加随机
            forest.fit(X, y)
            print('Sampling method %d, Forest index %d, Model accuracy using OOB %f' % (
            i, forest_count, forest.oob_score_))
            model_accuracy.append(forest.oob_score_)

            # Getting and saving OOB prediction
            data_balanced.loc[:, 'oob_class_pred'] = [forest.classes_[x] for x in
                                                      np.argmax(forest.oob_decision_function_, axis=1)]
            data_balanced.loc[:, 'class_%d' % forest.classes_[0]] = forest.oob_decision_function_[:, 0]  # 每个预测的值得多少
            data_balanced.loc[:, 'class_%d' % forest.classes_[1]] = forest.oob_decision_function_[:, 1]
            outfile = './ML-CamReady/OOBpred/OOBpredictions_%dt_%de_eIndex%d' % (
            forest_size, sum(models_reps), len(estimators_list))
            data_balanced.to_csv(outfile + '.csv')
            data_balanced.to_pickle(outfile + '.pkl')

            # Appending to estimators list
            estimators_list.append(('forest%d' % forest_count, forest))
            forest_count += 1

    # Saving Model Accuracy
    outfile = "./ML-CamReady/MA/" + 'OOBscores_%df_%dt_%de' % (
    len(features), forest_size, len(estimators_list))
    with open(outfile, 'w') as f:
        for oobscore in model_accuracy:
            f.write(str(oobscore) + '\n')

    # Getting prediction
    weights = None
    hard_vote_pred = np.asarray(
        [clf[1].predict(data_predict_orig.drop('ASN', axis=1)) for clf in estimators_list]).T  # 每个模型预测一个值
    hard_vote_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weights)), axis=1,
                                         arr=hard_vote_pred.astype('int'))

    # Adding Predictions
    data_predict_orig.loc[:, 'HardVotePred'] = hard_vote_pred
    print ('Hard Vote')
    print (data_predict_orig['HardVotePred'].value_counts())

    # Output to file
    outfile = './ML-CamReady/predictions_%df_%dt_%de' % (
    len(features), forest_size, len(estimators_list))
    data_predict_orig.to_csv(outfile + '.csv')
    data_predict_orig.to_pickle(outfile + '.pkl')


if __name__ == '__main__':
    main()
