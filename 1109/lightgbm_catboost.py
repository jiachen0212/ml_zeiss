# coding=utf-8
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as TTS
import pandas as pd
import numpy as np
from sklearn import metrics
import catboost as cb
 


X = np.load(r'./X_tf.npy')
Y = np.load(r'./Y_tf.npy')
y = np.array([[Y_[5]] for Y_ in Y])
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=668)


# # 创建成lgb特征的数据集格式
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
# # 将参数写成字典下形式
# params = {
#  'task': 'train',
#  'boosting_type': 'gbdt',  # 设置提升类型
#  'objective': 'regression',  # 目标函数
#  'metric': {'l2', 'auc'},  # 评估函数
#  'num_leaves': 31,  # 叶子节点数
#  'learning_rate': 0.05,  # 学习速率
#  'feature_fraction': 0.9,  # 建树的特征选择比例
#  'bagging_fraction': 0.8,  # 建树的样本采样比例
#  'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#  'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }
 
# # 训练 cv and train
# gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
 
# # 保存模型到文件
# #gbm.save_model('model.txt')
# joblib.dump(lgb, './model/lgb.pkl')
 
# # 预测数据集
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
 
# # 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

 
# 2. sklearn接口形式的Lightgbm

# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
 
# # 创建模型，训练模型
# gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
# gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
 
# # 测试机预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
 
# # 模型评估
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
 
# # feature importances
# print('Feature importances:', list(gbm.feature_importances_))
 
# # 网格搜索，参数优化
# estimator = lgb.LGBMRegressor(num_leaves=31)
# param_grid = {
#  'learning_rate': [0.01, 0.1, 1],
#  'n_estimators': [20, 40]
# }
# gbm = GridSearchCV(estimator, param_grid)
# gbm.fit(X_train, y_train)
# print('Best parameters found by grid search are:', gbm.best_params_)





######################## catboost ###########################

# 调参，用网格搜索调出最优参数
params = {'depth': [i for i in range(3, 14, 2)],
          'learning_rate': [i*0.01 for i in range(10)] + [0.2, 0.3],
          'l2_leaf_reg': [1, 4, 9],
          'iterations': [100, 300]}
cb = cb.CatBoostRegressor()
cb_model = GridSearchCV(cb, params, scoring="neg_log_loss", cv=3)
cb_model.fit(X_train, y_train)

# 查看最佳分数
print(cb_model.best_score_)   
# 查看最佳参数
print(cb_model.best_params_)   


# clf = cb.CatBoostRegressor(iterations=200,
#                              learning_rate=0.02,
#                              depth=12,
#                              eval_metric='RMSE',
#                              random_seed = 23,
#                              bagging_temperature = 0.2,
#                              od_type='Iter',
#                              metric_period = 75,
#                              od_wait=100)
# clf.fit(X_train, y_train)

# print(mean_squared_error(clf.predict(X_test), y_test))