# coding=utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import GridSearchCV


X = np.load(r'./X_tf.npy')
Y = np.load(r'./Y_tf.npy')


def lightgbm_hyperparameter_searching(X, Y, ind):
	# 就测试下500这个最难训的频段先..
	y = np.array([[Y_[ind]] for Y_ in Y])
	X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=668)

	# 参数搜索1
	# params = {
	# 'boosting_type': 'gbdt',
	# 'objective': 'regression',
	# 'metric': {'l2', 'auc'},
	# 'nthread':4,
	# 'learning_rate':0.1,
	# 'num_leaves':64,
	# 'max_depth': 9,
	# 'subsample': 0.7,
	# 'colsample_bytree': 0.5,
	# }
	# data_train = lgb.Dataset(X_train, y_train)
	# cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
	# print('best n_estimators:', len(cv_results['auc-mean']))
	# print('best cv score:', pd.Series(cv_results['auc-mean']).max())

	# 参数搜索2
	# params_test1={'max_depth': range(3,8,1), 'num_leaves':range(5, 100, 5)}
	# gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier
	# 	(boosting_type='gbdt',objective='regression',metrics={'l2', 'auc'},learning_rate=0.1, n_estimators=100, max_depth=6, bagging_fraction = 0.8,feature_fraction=0.8),param_grid = params_test1, scoring='neg_mean_squared_error',cv=5,n_jobs=-1)
	# gsearch1.fit(X_train,y_train)
	# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


	# 参数搜索3
	import optuna
	from optuna.integration import LightGBMPruningCallback
	def objective(trial, X, y):
		param_grid = {
			"n_estimators": trial.suggest_int("n_estimators", 60, 120, 10),
			"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
			"num_leaves": trial.suggest_int("num_leaves", 3, 20, step=1),
			"max_depth": trial.suggest_int("max_depth", 3, 12),
			"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 100, step=10),
			"max_bin": trial.suggest_int("max_bin", 100, 200),
			# "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
			"lambda_l2": trial.suggest_int("lambda_l2", 50, 100, step=5),
			"min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
			"bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8, step=0.1),
			"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
			"feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.8, step=0.1),
			"random_state": 666,
		}

		model = lgb.LGBMRegressor(objective="regression", **param_grid)
		model.fit(
			X_train,
			y_train,
			eval_set=[(X_test, y_test)],
			eval_metric="regression",
			early_stopping_rounds=100,
			callbacks=[
				LightGBMPruningCallback(trial, "l2")
			],
		)
		# 模型预测
		preds = model.predict(X_test)
		# 优化指标logloss最小
		res = mean_squared_error(y_test, preds)

		return res

	study = optuna.create_study(direction="minimize", study_name="LGBM Regression")
	func = lambda trial: objective(trial, X, y)
	study.optimize(func, n_trials=20)

lightgbm_hyperparameter_searching(X, Y, 5)

params = {'n_estimators': 70, 'learning_rate': 0.14026343390292878, 'num_leaves': 37, 'max_depth': 7, 'min_data_in_leaf': 74, 'max_bin': 260, 'lambda_l2': 65, 'min_gain_to_split': 4.75978070455136, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'feature_fraction': 0.8}
gbm = lgb.LGBMRegressor(objective='regression', **params)

# y = np.array([[Y_[5]] for Y_ in Y])
# X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=668)
# gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
#
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
#
# print('The mse of prediction is:', mean_squared_error(y_test, y_pred))
# feature importances
# print('Feature importances:', list(gbm.feature_importances_))
