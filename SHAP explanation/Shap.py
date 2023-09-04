import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
data1=pd.read_csv('Gra_feature.csv')
data2=pd.read_csv('data0.csv')
data3=pd.read_csv('data1.csv')
X1=data1.iloc[:,1:]
X2=data2.iloc[:,1:]
y=data3.iloc[:,-2]
Y=data3.iloc[0,-1:]
print(X2)
print(y)
print(Y)
shap.initjs()
estimator1 = GradientBoostingRegressor(learning_rate=0.1,max_depth=19,max_features=11,min_samples_leaf=6,n_estimators=819).fit(X1,y)
estimator2 = RandomForestRegressor(n_estimators=515,criterion='mse',max_depth=17,max_features=19,min_impurity_decrease=0.0,
                              random_state=19,n_jobs=1,min_impurity_split=2,min_samples_leaf=1).fit(X2,y)
# sel = VarianceThreshold(threshold=0.09)
# X=pd.DataFrame(sel.fit_transform(X))
# print(X)
explainer1 = shap.Explainer(estimator1, X1)
explainer2 = shap.Explainer(estimator2, X2)
# print(X[1:-1])
shap_values1 = explainer1(X1[:-1],check_additivity=False)
shap_values2 = explainer2(X2[:-1],check_additivity=False)
plt.xticks(fontsize=1)
plt.yticks(fontsize=0.1)
# shap.plots.heatmap(shap_values1,max_display=14,
#                             feature_values=shap_values1.abs.max(0))
# shap.plots.heatmap(shap_values2,max_display=14,
#                             feature_values=shap_values2.abs.max(0))
# shap.plots.bar(shap_values, max_display=15)#
# shap.plots.waterfall(shap_values1[310],max_display=19)
# shap.plots.waterfall(shap_values2[64],max_display=19)
# shap.initjs()
# shap.force_plot(shap_values1)
shap.force_plot(shap_values2[5])
# shap.plots.beeswarm(shap_values1,max_display=20)
shap.plots.beeswarm(shap_values1,max_display=11)
shap.plots.beeswarm(shap_values2,max_display=11)
shap.plots.bar(shap_values1,max_display=11)
shap.plots.bar(shap_values2,max_display=11)
# clust = shap.utils.hclust(X, y, linkage="single")
# shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)
