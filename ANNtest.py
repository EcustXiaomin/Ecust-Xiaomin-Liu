import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
data1 = pd.read_excel('solubility_6_8_10.xlsx',sheet_name='Sheet1')
data2 = pd.read_excel('solubility_6_8_10.xlsx',sheet_name='Sheet2')
data3 = pd.read_excel('solubility_6_8_10.xlsx',sheet_name='Sheet3')
scaler = MinMaxScaler()
X1=data1.iloc[:,1:9]
X2=data2.iloc[:,1:11]
X3=data3.iloc[:,1:13]
X1 = scaler.fit_transform(X1)
X1 = pd.DataFrame(X1)
X2 = scaler.fit_transform(X2)
X2 = pd.DataFrame(X2)
X3 = scaler.fit_transform(X3)
X3 = pd.DataFrame(X3)
pre = pd.DataFrame()
exp = pd.DataFrame()
alpha = np.arange(0.0001, 0.01, 0.0001)
Y = pd.DataFrame(data1.iloc[:, 13])
b=[]
X1train, X1test, Y1train, Y1test = train_test_split(X1,Y,test_size=0.2,random_state=0)
superpa1 = []
Y1train=np.ravel(Y1train)
for i in tqdm(alpha, desc='Processing'):
    regressor1 = MLPRegressor(hidden_layer_sizes=[100, 100], alpha=i, max_iter=1000)
    a1 = cross_val_score(regressor1, X1train, Y1train, cv=10, scoring="neg_mean_absolute_error")
    c1 = cross_val_score(regressor1, X1train, Y1train, cv=10, scoring='r2')
    superpa1.append(c1.mean())
print(max(superpa1), superpa1.index(max(superpa1)))
p1 = int(superpa1.index(max(superpa1)))
regressor1 = MLPRegressor(hidden_layer_sizes=[100, 100], alpha=(p1 + 1) * 0.0001, max_iter=1000,verbose=True,validation_fraction=0.2,early_stopping=True)
Hcmodel1 = regressor1.fit(X1train, Y1train)
Hcpre1 = Hcmodel1.predict(X1test)
Hcpre1 = pd.DataFrame(Hcpre1)
pre1 = pd.concat([pre, Hcpre1], axis=0, ignore_index=True)
exp1 = pd.concat([exp, Y1test], axis=0, ignore_index=True)
b1=r2_score(Y1test,Hcpre1)
b.append(b1)
train_loss = Hcmodel1.loss_curve_
val_loss = regressor1.validation_scores_
plt.plot(train_loss,label='Trainingloss')
plt.plot(val_loss,label='validationloss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()