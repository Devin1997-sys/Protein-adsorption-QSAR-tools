from sklearn import preprocessing
import pandas as pd
data1=pd.read_csv('all-adsorption_PRO+MOL.csv')
X=data1.iloc[:,3:-1]
y=data1.iloc[:,-1:]
scaler = preprocessing.StandardScaler().fit(X)
print(scaler)
print(scaler.mean_)
print(scaler.scale_)
X_scaled = scaler.transform(X)
print(X_scaled)
data = pd.DataFrame(X_scaled)
data.to_csv('h.csv')