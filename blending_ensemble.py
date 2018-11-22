import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

def blending(train_x, train_y, result_x, clfs):
    x_train, x_dev, y_train, y_dev = train_test_split(train_x, 
                                                    train_y, 
                                                    test_size=0.4, 
                                                    random_state=0)
    x_test1, x_test2, y_test1, y_test2 = train_test_split(x_dev,
                                                          y_dev,
                                                          test_size=0.5,
                                                          random_state=0)
    
    blend_test1_i = np.zeros([x_test1.shape[0], len(clfs[0])])
    blend_test2_i = np.zeros([x_test2.shape[0], len(clfs[0])])
    blend_result_i = np.zeros([result_x.shape[0], len(clfs[0])])
    for i, clf in enumerate(clfs[0]):
        clf.fit(x_train, y_train)
        blend_test1_i[:, i] = clf.predict_proba(x_test1)[:, 1]
        blend_test2_i[:, i] = clf.predict_proba(x_test2)[:, 1]
        print("model {}, Auc (val): {:.4f}".format(i, metrics.roc_auc_score(y_test1, blend_test1_i[:, i])))
        blend_result_i[:, i] = clf.predict_proba(result_x)[:, 1]
    
    blend_test2 = np.zeros([x_test2.shape[0], len(clfs[1])]) 
    result = np.zeros([result_x.shape[0], len(clfs[1])]) 
    for i, clf in enumerate(clfs[1]):
        clf.fit(blend_test1_i, y_test1)
        blend_test2[:, i] = clf.predict_proba(blend_test2_i)[:, 1]
        print("level2 model {}, Auc (val): {:.4f}".format(i, metrics.roc_auc_score(y_test2, blend_test2[:, i])))
        result[:, i] = clf.predict_proba(blend_result_i)[:, 1]
    
    print("final Auc", metrics.roc_auc_score(y_test2, blend_test2.mean(axis=1)))    
    final_result = result.mean(axis=1)
    return final_result

def load_data():
    dataset1 = pd.read_csv("data/dataset1.csv")
    dataset2 = pd.read_csv("data/dataset2.csv")
    dataset3 = pd.read_csv("data/dataset3.csv")
     
    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)
    
    dataset1.label.replace(-1, 0, inplace=True)
    dataset2.label.replace(-1, 0, inplace=True)
    
    dataset1.fillna(999, inplace=True)
    dataset2.fillna(999, inplace=True)
    dataset3.fillna(999, inplace=True)
    
    dataset1_x = dataset1.drop(["user_id", "label", "day_gap_before", "day_gap_after"], axis=1)
    dataset1_y = dataset1.label
    dataset2_x = dataset2.drop(["user_id", "label", "day_gap_before", "day_gap_after"], axis=1)
    dataset2_y = dataset2.label
    
    dataset12 = pd.concat([dataset1, dataset2], axis=0)
    dataset12_x = dataset12.drop(["user_id", "label", "day_gap_before", "day_gap_after"], axis=1)
    dataset12_y = dataset12.label
    
    dataset3_preds = dataset3[["user_id", "coupon_id", "date_received"]]
    dataset3_x = dataset3.drop(["user_id", "coupon_id", "date_received", "day_gap_before", "day_gap_after"], axis=1)
    
    print(dataset1_x.shape, dataset2_x.shape, dataset3_x.shape)
    return dataset12_x, dataset12_y, dataset3_x

def load_gbdt_models():
    params1 = { "n_estimators":100,         
               "loss":"deviance",      
               "learning_rate":0.1, 
               "max_depth":5,           
               "subsample":0.5,     
               # "min_samples_split":2,
               "min_samples_leaf":100,    
               "max_features":None,        
               "max_leaf_nodes":None,
               "verbose":0,                    
               "random_state":0
    }
    gbdt1 = GradientBoostingClassifier(**params1)
    return [gbdt1]

def load_rf_models():
    params1 = { "n_estimators":200,         
               "max_depth":7,           
               # "min_samples_split":2,    
               "min_samples_leaf":150,    
               "max_features":None,        
               "max_leaf_nodes":None,    
               "n_jobs":-1,
               "verbose":0,                    
               "random_state":0
    }
    rf1 = RandomForestClassifier(**params1)
    return [rf1]

def load_xgb_models():
    params1 = {"objective":"rank:pairwise",
              "gamma":0.1,
              "min_child_weight":1.1,
              "max_depth":5,
              "reg_lambda": 10,
              "subsample":0.7,
              "colsample_bytree":0.7,
              "colsample_bylevel":0.7,
              "learning_rate":0.01,
              "seed":0,
        }
    xgb_model1 = xgb.XGBClassifier(**params1)
    return [xgb_model1]

def load_gbm_models():
    params1 = {'learning_rate':0.05, 
              'boosting_type':'gbdt', 
              'objective':'binary', 
              'metric':'auc',
              'max_depth':5, 
              'num_leaves':32, 
              'feature_fraction':0.6, 
              'min_data_in_leaf':300,
              'bagging_fraction':0.6, 
              'bagging_freq':5,
              'seed':0,
              "is_unbalance":True,
              "verbose":-1} 
    gbm1 = lgb.LGBMClassifier(**params1)
    return [gbm1]

def load_models(level):
    if level == 1:
        return load_gbdt_models() + load_rf_models() + load_xgb_models() + load_gbm_models()
    elif level == 2:
        return [LogisticRegression(), xgb.XGBClassifier()]
    
def main():
    dataset12_x, dataset12_y, dataset3_x = load_data()
    clfs = [load_models(1), load_models(2)]
    result = blending(dataset12_x, dataset12_y, dataset3_x, clfs)


if __name__ == "__main__":
    main()