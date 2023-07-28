import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
import math

##数据导入
data1=pd.read_csv('打乱后特征.csv')
X=data1.iloc[:,2:-3]
y=data1.iloc[:,-1:]
print(X)

##特征选择
sel = VarianceThreshold(threshold=0.09)
X=sel.fit_transform(X)
print(X)
print(X.shape)
estimator = RandomForestRegressor(n_estimators=515,criterion='mse',
                                  max_depth=17,min_impurity_decrease=0.0,
                            random_state=19,n_jobs=1,min_samples_leaf=1)
selector = RFE(estimator,step=0.1,n_features_to_select=30)
selector.fit(X,y.values.ravel())
print("rank：",selector.ranking_)
data = selector.ranking_
X1=pd.DataFrame()
print(X1)

if X1.empty == True:
    X1 = pd.DataFrame(data1.iloc[:, 2])
print(X1)
print(X1.shape)
for index,i in enumerate(data):
        if i == 1:
            print(index)
            X2=pd.DataFrame(data1.iloc[:,index+2])
            X1= X1.join(X2)
print(X1)

##关键特征处理
outputpath='C:\\Users\yons\model\qsar supplement/data4-2-RFdescriptors.csv'
X1.to_csv(outputpath,sep=',',index=False,header=True)
data2=pd.read_csv('data4-2-RFdescriptors.csv')
X3=data2.iloc[:,0:]
X3=np.array(X3)
print(X3.shape[1])

##训练集和测试集划分
from sklearn.model_selection import train_test_split
for random_state_1 in range(1,11):
    X_train,X_test,y_train,y_test=train_test_split(X3,y,
                                                 test_size=0.1,
                                                random_state=random_state_1)
    model = RandomForestRegressor(n_estimators=515,criterion='mse',
                                  max_depth=17,min_impurity_decrease=0.0,
                              random_state=19,n_jobs=1,min_samples_leaf=1)
    model.fit(X_train,y_train)
    pred_test = model.predict(X_test)
    score = explained_variance_score(y_test,pred_test)
    score1=r2_score(y_test,pred_test)
    score3=mean_squared_error(y_test,pred_test)
    score3 =math.sqrt(score3)
    print('r2 test', score1)
    print('rmse test',score3)

##交叉验证
kf = KFold(n_splits=10,shuffle=True,random_state=1)
for train_index, validation_index in kf.split(X_train):
    print("Train:", train_index, "Validation:",validation_index)
    X_train_train, X_validation = X_train[train_index], X_train[validation_index]
    y_train_train, y_validation = y_train.iloc[train_index,:], y_train.iloc[validation_index,:]
i = 1
all,g,m,j,r=0,0,0,0,0
for train_index,validation_index in kf.split(X_train, y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    X_train_train, X_validation = X_train[train_index], X_train[validation_index]
    y_train_train, y_validation = y_train.iloc[train_index,:], y_train.iloc[validation_index,:]
    model = RandomForestRegressor()
    model.fit(X_train_train,y_train_train)
    pred_validation=model.predict(X_validation)
    score = explained_variance_score(y_validation, pred_validation)
    score2 = r2_score(y_validation, pred_validation)
    score3_1 = mean_squared_error(y_validation,pred_validation)
    score3_1 = math.sqrt(score3_1)
    print('explained_variance_score', score)
    print('r2_score', score2)
    print('rmse',score3_1)
    all = score2 +all
    g = score3_1 + g
    print('average r2_validation',all/i)
    print('average rmse validation',g/i)
    pred_yTrain = model.predict(X_train_train)
    score3_2 = mean_squared_error(y_train_train,pred_yTrain)
    score3_2 = math.sqrt(score3_2)
    print('rmse_train',score3_2)
    score4 = explained_variance_score(y_train_train, pred_yTrain)
    r = score4 + r
    m = score3_2 + m
    print('average r2_train', r/i)
    print('average rmse train',m/i)
    i += 1

#模型评价指标
print('r2 test', score1)
print('rmse test',score3)

#数据可视化
fig,axes = plt.subplots(figsize=(6,5))
x1_label = axes.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = axes.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
plt.scatter(pred_yTrain,y_train_train,
             c='dimgray',
             edgecolor='dimgray',
             marker='.',
             s=100,
             alpha=0.9,
             label='Training Data')
plt.scatter(pred_validation,y_validation,
            c='lightcoral',
            edgecolor='lightcoral',
            marker='s',s=30,
            alpha=0.9,
            label='Validation Data')
plt.scatter(pred_test,y_test,c='cornflowerblue',
             edgecolor='cornflowerblue',
             marker='^',
             s=30,
             alpha=0.9,
             label='Test Data')
plt.xlim([0,1])
a=np.arange(0,1.1,0.1)
plt.plot(a,a,'k--')
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.xlabel("Predicted q*",fontsize=18)
plt.ylabel("Experimental q*",fontsize=18)
plt.show()
# plt.legend(['Train R2=0.94 RMSE=3.98','Validation R2=0.89 RMSE=5.57\n10-fold Cross validation\nR2=0.70 RMSE=8.59']
#            , fontsize = 8, prop = 'Times New Roman',labelspacing=1,frameon=False,
#            ncol = 1,loc=0,
#            bbox_to_anchor = (0.5,1))


# ax = plt.subplot(1,2,2)
#
# # plt.scatter(pred_yTrain,y_train_train,
# #              c='gray',
# #              edgecolor='gray',
# #              marker='.',
# #              s=100,
# #              alpha=0.9,
# #              label='Training Data')
#
# plt.scatter(pred_test,y_test,c='orange',
#              edgecolor='orange',
#              marker='^',
#              s=50,
#              alpha=0.9,
#              label='Test Data')
#
# plt.xlabel("Predicted q*",fontsize=18)
# plt.ylabel("Experimental q*",fontsize=18)
#
# # plt.legend(['Train R2=0.94 RMSE=3.98','Test R2=0.79 RMSE=7.14'], fontsize = 8,
# #            ncol = 1,loc=0, prop = 'Times New Roman',labelspacing=1,frameon=False,
# #            bbox_to_anchor = (0.5,1))
#
# plt.xlim([0,50])
# a=np.arange(0,60,10)
# plt.plot(a,a,'k--')
# plt.xticks(size = 18)
# plt.yticks(size = 18)


# plt.plot(a,a+0.5,'k--')
# plt.plot(a,a-0.5,'k--')
# plt.grid()



