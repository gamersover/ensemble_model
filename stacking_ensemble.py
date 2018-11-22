#-*- coding:utf-8 -*-
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
import numpy as np
import pandas as pd
from utils import DataHelper

def get_loss(y, y_pre):
    loss = (y - y_pre)**2
    loss = loss.sum() / (y.size*2)
    return loss

def sub_result(result, type=''):
    submission = pd.DataFrame({'pred':result})
    submission.to_csv(r'../results/submission_{}.csv'.format(type), header=None, 
                      index=False, float_format='%.4f')
    print('结果保存在results/submission_{}.csv'.format(type))

def run(train_x, train_y, result_x, clfs, K):
    x_dev, x_test, y_dev, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    ss = KFold(n_splits=K, shuffle=True)

    blend_train = np.zeros((x_dev.shape[0], len(clfs)))
    blend_test = np.zeros((x_test.shape[0], len(clfs)))
    blend_result = np.zeros((result_x.shape[0], len(clfs)))
    print('x_test shape', x_test.shape)
    print('blend_train shape', blend_train.shape)
    print('blend_test shape', blend_test.shape)

    for i, clf in enumerate(clfs):
        print('Traning classifier [%s]' % (i))
        blend_test_i = np.zeros((x_test.shape[0], K))
        blend_result_i = np.zeros((result_x.shape[0], K))
        for j, (train_index, cv_index) in enumerate(ss.split(x_dev)):
            print('Fold [%s]' %(j))
            x_train, y_train = x_dev[train_index], y_dev[train_index]
            x_cv, y_cv = x_dev[cv_index], y_dev[cv_index]

            clf.fit(x_train, y_train)

            blend_train[cv_index, i] = clf.predict(x_cv)
            blend_test_i[:,j] = clf.predict(x_test)
            blend_result_i[:,j] = clf.predict(result_x)

        blend_test[:,i] = blend_test_i.mean(1)
        print('model [%s  %s]' %(i, get_loss(y_test, blend_test[:,i])))
        blend_result[:,i] = blend_result_i.mean(1)
        
    print('result shape', blend_result.shape)
#     blend = pd.DataFrame(blend_result)
#     blend.to_csv('../result_analyze/blend.csv', header=None, index=False)
    bclf = LinearRegression()
    bclf.fit(blend_train, y_dev)
    y_test_pred = bclf.predict(blend_test)
    result = bclf.predict(blend_result)
    print('test loss', get_loss(y_test, y_test_pred))
    return result

if __name__ == '__main__':
    datahelper = DataHelper(ftype='median', sex_split=True)
    train_x, train_y, test_x, feat = datahelper.load_data()
    
    sds = StandardScaler()
#     mms = MinMaxScaler()
#     trian_x = sds.fit_transform(train_x)
#     test_x = sds.fit_transform(test_x)
    
    rfr = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    etr = ExtraTreesRegressor()
    gbmr = LGBMRegressor()
#                     boosting_type='gbdt',
#                     objective='regression',
#                     metric='mse',
#                     feature_fraction=0.8,
#                     num_boost_round=3000,
#                     learning_rate=0.01,
#                     max_depth=7,
#                     num_leaves=40,
#                     min_data_in_leaf=100,
#                     bagging_fraction=0.8,
#                     bagging_freq=4)
    xgbr = XGBRegressor()
#     adbr = AdaBoostRegressor(n_estimators=100, learning_rate=0.02)
    clfs = [rfr, gbr, etr, gbmr, xgbr]
    result = run(train_x, train_y, test_x, clfs, K=5)
    sub_result(result, 'stacking5-59')