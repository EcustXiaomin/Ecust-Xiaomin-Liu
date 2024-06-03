import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
data = pd.read_excel('solubility-types.xlsx',sheet_name='Sheet3')
X=data.iloc[:,2:14]
X = pd.DataFrame(X)
Y = pd.DataFrame(data.iloc[:, -1])
alpha = list(np.arange(0.01, 0.11, 0.01))
deep = list(np.arange(1,11))
Xvalid = pd.DataFrame()
Xpre_new = pd.DataFrame()
Xtrain=data.iloc[:,2:14]
Ytrain=data.iloc[:,-1]
superpa = []
kf = KFold(n_splits=10, shuffle=True)
for D in deep:
    for Q in alpha:
        regressor = XGBRegressor(max_depth=D, learning_rate=Q, n_estimators=200, objective='reg:squarederror',
                                 booster='gbtree', gamma=0)
        b1 = cross_val_score(regressor, Xtrain, Ytrain, cv=kf, scoring='r2')
        superpa.append(b1.mean())
p1 = int(superpa.index(max(superpa)))
print(p1)
regressor = XGBRegressor(max_depth=(p1//10)+1, learning_rate=alpha[p1%10], n_estimators=200, objective='reg:squarederror',
                     booster='gbtree', gamma=0)
Hcmodel = regressor.fit(Xtrain, Ytrain)