#基本工具
import numpy as np
import pandas as pd
import time
#算法/损失/评估指标等
import sklearn
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from bayes_opt import BayesianOptimization
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
import optuna

data1=pd.read_csv('Gra_feature.csv')
y=data1.iloc[:,-1:]
outputpath='C:\\Users\yons\model\qsar supplement/Data-all-GBDTdescriptors.csv'
data2=pd.read_csv('Data-all-GBDTdescriptors.csv')
X=data2.iloc[:,1:]
X=np.array(X)
#max_features, min_impurity_decrease,
def bayesopt_objective(n_estimators, max_depth,learning_rate, min_samples_leaf,min_samples_split):
    # 定义评估器
    # 需要调整的超参数等于目标函数的输入，不需要调整的超参数则直接等于固定值
    # 默认参数输入一定是浮点数，因此需要套上int函数处理成整数
    # reg = RFR(n_estimators=int(n_estimators)
    #           , max_depth=int(max_depth)
    #           , max_features=int(max_features)
    #           , min_impurity_decrease=min_impurity_decrease
    #           , min_samples_leaf=int(min_samples_leaf)
    #           , random_state=19
    #           , min_samples_split= int(min_samples_split)
    #           , verbose=False  # 可自行决定是否开启森林建树的verbose
    #           )
    reg = GBDT(n_estimators=int(n_estimators)
               ,learning_rate=learning_rate
               ,min_samples_leaf=int(min_samples_leaf)
               , min_samples_split= int(min_samples_split)
               , max_depth=int(max_depth)

               )
    # 定义损失的输出，5折交叉验证下的结果，输出负根均方误差（-RMSE）
    # 注意，交叉验证需要使用数据，但我们不能让数据X,y成为目标函数的输入
    cv = KFold(n_splits=10, shuffle=True, random_state=19)
    validation_loss = cross_validate(reg, X, y
                                     , scoring="neg_root_mean_squared_error"
                                     , cv=cv
                                     , verbose=False
                                     , n_jobs=-1
                                     , error_score='raise'
                                     # 如果交叉验证中的算法执行报错，则告诉我们错误的理由
                                     )

    # 交叉验证输出的评估指标是负根均方误差，因此本来就是负的损失
    # 目标函数可直接输出该损失的均值
    return np.mean(validation_loss["test_score"])


param_grid_simple = {'n_estimators': (100,1000)
                     , 'max_depth':(10,25)
                     , "learning_rate": (0.0001,0.1)

                     ,'min_samples_leaf':(1,10)

                     ,'min_samples_split':(2,10)
                    }
#"min_impurity_decrease":(0,1) "alpha":(0,1),"verbose":(0,1)
def param_bayes_opt(init_points, n_iter):
    # 定义优化器，先实例化优化器
    opt = BayesianOptimization(bayesopt_objective  # 需要优化的目标函数
                               , param_grid_simple  # 备选参数空间
                               , random_state=19  # 随机数种子，虽然无法控制住
                               )
    opt.maximize(init_points=init_points  # 抽取多少个初始观测值
                 , n_iter=n_iter  # 一共观测/迭代多少次
                 )

    # 优化完成，取出最佳参数与最佳分数
    params_best = opt.max["params"]
    score_best = opt.max["target"]

    # 打印最佳参数与最佳分数
    print("\n", "\n", "best params: ", params_best,
          "\n", "\n", "best cvscore: ", score_best)

    # 返回最佳参数与最佳分数
    return params_best, score_best

start = time.time()
params_best, score_best = param_bayes_opt(1,10) #初始看30个观测值，后面迭代280次
print('It takes %s minutes' % ((time.time() - start)/60))
validation_score = bayes_opt_validation(params_best)
print("\n","\n","validation_score: ",validation_score)
