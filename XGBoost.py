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
from sklearn.metrics import mean_squared_error
XGBpre_new1 = pd.DataFrame()
XGBtra_new1 = pd.DataFrame()
XGBpre_new2 = pd.DataFrame()
XGBtra_new2 = pd.DataFrame()
XGBpre_new3 = pd.DataFrame()
XGBtra_new3 = pd.DataFrame()
XGBpre_new4 = pd.DataFrame()
XGBtra_new4 = pd.DataFrame()
XGBexp_new1 = pd.DataFrame()
XGBexp_new2 = pd.DataFrame()
XGBexp_new3 = pd.DataFrame()
XGBexp_new4 = pd.DataFrame()
XGBexp_new11 = pd.DataFrame()
XGBexp_new22 = pd.DataFrame()
XGBexp_new33 = pd.DataFrame()
XGBexp_new44 = pd.DataFrame()
deep = list(np.arange(1,11))
alpha = list(np.arange(0.01, 0.11, 0.01))
P1=[]
P2=[]
P3=[]
P4=[]
R1=[]
R2=[]
R3=[]
R4=[]
save_directory1 = "6regions-cross"
save_directory2= "10regions-cross"
save_directory3 = "3regions-cross"
save_directory4 = "12regions-cross"
for fold in tqdm(range(1, 11),desc='Processing'):
    train_data1 = pd.read_csv(f"{save_directory1}fold_{fold}_train.csv")
    test_data1 = pd.read_csv(f"{save_directory1}fold_{fold}_test.csv")
    train_data2 = pd.read_csv(f"{save_directory2}fold_{fold}_train.csv")
    test_data2 = pd.read_csv(f"{save_directory2}fold_{fold}_test.csv")
    train_data3 = pd.read_csv(f"{save_directory3}fold_{fold}_train.csv")
    test_data3 = pd.read_csv(f"{save_directory3}fold_{fold}_test.csv")
    train_data4 = pd.read_csv(f"{save_directory4}fold_{fold}_train.csv")
    test_data4 = pd.read_csv(f"{save_directory4}fold_{fold}_test.csv")
    superpa1 = []
    superpa2 = []
    superpa3 = []
    superpa4 = []
    X_cross1 = train_data1.iloc[:, 2:10]
    X_cross2 = train_data2.iloc[:, 2:14]
    X_cross3 = train_data3.iloc[:, 2:7]
    X_cross4 = train_data4.iloc[:, 2:16]
    print(X_cross1)
    Y_cross1 = train_data1.iloc[:, -1]
    Y_cross2 = train_data2.iloc[:, -1]
    Y_cross3 = train_data3.iloc[:, -1]
    Y_cross4 = train_data4.iloc[:, -1]
    Xtest_1 = test_data1.iloc[:, 2:10]
    Xtest_2 = test_data2.iloc[:, 2:14]
    Xtest_3 = test_data3.iloc[:, 2:7]
    Xtest_4 = test_data4.iloc[:, 2:16]
    Ytest_1 = test_data1.iloc[:, -1]
    Ytest_2 = test_data2.iloc[:, -1]
    Ytest_3 = test_data3.iloc[:, -1]
    Ytest_4 = test_data4.iloc[:, -1]
    kf = KFold(n_splits=10, shuffle=True)
    for D in tqdm(deep):
        for Q in tqdm(alpha):
            regressor = XGBRegressor(max_depth=D, learning_rate=Q, n_estimators=200, objective='reg:squarederror',
                                     booster='gbtree', gamma=0)
            b1 = cross_val_score(regressor, X_cross1, Y_cross1, cv=kf, scoring='r2')
            b2 = cross_val_score(regressor, X_cross2, Y_cross2, cv=kf, scoring='r2')
            b3 = cross_val_score(regressor, X_cross3, Y_cross3, cv=kf, scoring='r2')
            b4 = cross_val_score(regressor, X_cross4, Y_cross4, cv=kf, scoring='r2')
            superpa1.append(b1.mean())
            superpa2.append(b2.mean())
            superpa3.append(b3.mean())
            superpa4.append(b4.mean())
    p1 = int(superpa1.index(max(superpa1)))
    p2 = int(superpa2.index(max(superpa2)))
    p3 = int(superpa3.index(max(superpa3)))
    p4 = int(superpa4.index(max(superpa4)))
    R1.append(max(superpa1))
    R2.append(max(superpa2))
    R3.append(max(superpa3))
    R4.append(max(superpa4))
    P1.append(p1)
    P2.append(p2)
    P3.append(p3)
    P4.append(p4)
    regressor1 = XGBRegressor(max_depth=p1 // 10 + 1, learning_rate=alpha[p1 % 10], n_estimators=200,
                              objective='reg:squarederror',
                              booster='gbtree', gamma=0)
    regressor2 = XGBRegressor(max_depth=p2 // 10 + 1, learning_rate=alpha[p2 % 10], n_estimators=200,
                              objective='reg:squarederror',
                              booster='gbtree', gamma=0)
    regressor3 = XGBRegressor(max_depth=p3 // 10 + 1, learning_rate=alpha[p3 % 10], n_estimators=200,
                              objective='reg:squarederror',
                              booster='gbtree', gamma=0)
    regressor4 = XGBRegressor(max_depth=p4 // 10 + 1, learning_rate=alpha[p4 % 10], n_estimators=200,
                              objective='reg:squarederror',
                              booster='gbtree', gamma=0)
    Hcmodel1 = regressor1.fit(X_cross1, Y_cross1)
    Hctrain1 = Hcmodel1.predict(X_cross1)
    Hcpre1 = Hcmodel1.predict(Xtest_1)
    Hcpre1 = pd.DataFrame(Hcpre1)
    Hctrain1 = pd.DataFrame(Hctrain1)
    Hcmodel2 = regressor2.fit(X_cross2, Y_cross2)
    Hctrain2 = Hcmodel2.predict(X_cross2)
    Hcpre2 = Hcmodel2.predict(Xtest_2)
    Hcpre2 = pd.DataFrame(Hcpre2)
    Hctrain2 = pd.DataFrame(Hctrain2)
    Hcmodel3 = regressor3.fit(X_cross3, Y_cross3)
    Hctrain3 = Hcmodel3.predict(X_cross3)
    Hcpre3 = Hcmodel3.predict(Xtest_3)
    Hcpre3 = pd.DataFrame(Hcpre3)
    Hctrain3 = pd.DataFrame(Hctrain3)
    Hcmodel4 = regressor4.fit(X_cross4, Y_cross4)
    Hctrain4 = Hcmodel4.predict(X_cross4)
    Hcpre4 = Hcmodel4.predict(Xtest_4)
    Hcpre4 = pd.DataFrame(Hcpre4)
    Hctrain4 = pd.DataFrame(Hctrain4)
    XGBpre_new1 = pd.concat([XGBpre_new1, Hcpre1], axis=0, ignore_index=True)
    XGBexp_new1 = pd.concat([XGBexp_new1, Ytest_1], axis=0, ignore_index=True)
    XGBtra_new1 = pd.concat([XGBtra_new1, Hctrain1], axis=0, ignore_index=True)
    XGBexp_new11 = pd.concat([XGBexp_new11, Y_cross1], axis=0, ignore_index=True)
    XGBpre_new2 = pd.concat([XGBpre_new2, Hcpre2], axis=0, ignore_index=True)
    XGBexp_new2 = pd.concat([XGBexp_new2, Ytest_2], axis=0, ignore_index=True)
    XGBtra_new2 = pd.concat([XGBtra_new2, Hctrain2], axis=0, ignore_index=True)
    XGBexp_new22 = pd.concat([XGBexp_new22, Y_cross2], axis=0, ignore_index=True)
    XGBpre_new3 = pd.concat([XGBpre_new3, Hcpre3], axis=0, ignore_index=True)
    XGBexp_new3 = pd.concat([XGBexp_new3, Ytest_3], axis=0, ignore_index=True)
    XGBtra_new3 = pd.concat([XGBtra_new3, Hctrain3], axis=0, ignore_index=True)
    XGBexp_new33 = pd.concat([XGBexp_new33, Y_cross3], axis=0, ignore_index=True)
    XGBpre_new4 = pd.concat([XGBpre_new4, Hcpre4], axis=0, ignore_index=True)
    XGBexp_new4 = pd.concat([XGBexp_new4, Ytest_4], axis=0, ignore_index=True)
    XGBtra_new4 = pd.concat([XGBtra_new4, Hctrain4], axis=0, ignore_index=True)
    XGBexp_new44 = pd.concat([XGBexp_new44, Y_cross4], axis=0, ignore_index=True)
with pd.ExcelWriter('XGBpre_new.xlsx') as writer:
    pd.DataFrame(XGBpre_new1).to_excel(writer, sheet_name='6', index=False)
    pd.DataFrame(XGBpre_new2).to_excel(writer, sheet_name='10', index=False)
    pd.DataFrame(XGBpre_new3).to_excel(writer, sheet_name='3', index=False)
    pd.DataFrame(XGBpre_new4).to_excel(writer, sheet_name='12', index=False)
with pd.ExcelWriter('XGBexp_new.xlsx') as writer:
    pd.DataFrame(XGBexp_new1).to_excel(writer, sheet_name='6', index=False)
    pd.DataFrame(XGBexp_new2).to_excel(writer, sheet_name='10', index=False)
    pd.DataFrame(XGBexp_new3).to_excel(writer, sheet_name='3', index=False)
    pd.DataFrame(XGBexp_new4).to_excel(writer, sheet_name='12', index=False)
with pd.ExcelWriter('XGBtra_new.xlsx') as writer2:
    pd.DataFrame(XGBtra_new1).to_excel(writer2, sheet_name='6', index=False)
    pd.DataFrame(XGBtra_new2).to_excel(writer2, sheet_name='10', index=False)
    pd.DataFrame(XGBtra_new3).to_excel(writer2, sheet_name='3', index=False)
    pd.DataFrame(XGBtra_new4).to_excel(writer2, sheet_name='12', index=False)
with pd.ExcelWriter('XGBexp_new11.xlsx') as writer:
    pd.DataFrame(XGBexp_new11).to_excel(writer, sheet_name='6', index=False)
    pd.DataFrame(XGBexp_new22).to_excel(writer, sheet_name='10', index=False)
    pd.DataFrame(XGBexp_new33).to_excel(writer, sheet_name='3', index=False)
    pd.DataFrame(XGBexp_new44).to_excel(writer, sheet_name='12', index=False)
with pd.ExcelWriter('XGBindex.xlsx') as index:
    pd.DataFrame(P1).to_excel(index, sheet_name='6', index=False)
    pd.DataFrame(P2).to_excel(index, sheet_name='10', index=False)
    pd.DataFrame(P3).to_excel(index, sheet_name='3', index=False)
    pd.DataFrame(P4).to_excel(index, sheet_name='12', index=False)
with pd.ExcelWriter('XGBvalidR2.xlsx') as index:
    pd.DataFrame(R1).to_excel(index, sheet_name='6', index=False)
    pd.DataFrame(R2).to_excel(index, sheet_name='10', index=False)
    pd.DataFrame(R3).to_excel(index, sheet_name='3', index=False)
    pd.DataFrame(R4).to_excel(index, sheet_name='12', index=False)
XGBpre1 = pd.read_excel('XGBpre_new.xlsx', sheet_name='6')
XGBpre2 = pd.read_excel('XGBpre_new.xlsx', sheet_name='10')
XGBpre3 = pd.read_excel('XGBpre_new.xlsx', sheet_name='3')
XGBpre4 = pd.read_excel('XGBpre_new.xlsx', sheet_name='12')
XGBexp1 = pd.read_excel('XGBexp_new.xlsx', sheet_name='6')
XGBexp2 = pd.read_excel('XGBexp_new.xlsx', sheet_name='10')
XGBexp3 = pd.read_excel('XGBexp_new.xlsx', sheet_name='3')
XGBexp4 = pd.read_excel('XGBexp_new.xlsx', sheet_name='12')
XGBtra1 = pd.read_excel('XGBtra_new.xlsx', sheet_name='6')
XGBtra2 = pd.read_excel('XGBtra_new.xlsx', sheet_name='10')
XGBtra3 = pd.read_excel('XGBtra_new.xlsx', sheet_name='3')
XGBtra4 = pd.read_excel('XGBtra_new.xlsx', sheet_name='12')
XGBexp11 = pd.read_excel('XGBexp_new11.xlsx', sheet_name='6')
XGBexp22 = pd.read_excel('XGBexp_new11.xlsx', sheet_name='10')
XGBexp33 = pd.read_excel('XGBexp_new11.xlsx', sheet_name='3')
XGBexp44 = pd.read_excel('XGBexp_new11.xlsx', sheet_name='12')
b1XGBpre = r2_score(XGBexp1,XGBpre1)
b2XGBpre = r2_score(XGBexp2,XGBpre2)
b3XGBpre = r2_score(XGBexp3,XGBpre3)
b4XGBpre = r2_score(XGBexp4,XGBpre4)
b1XGBtra = r2_score(XGBexp11,XGBtra1)
b2XGBtra = r2_score(XGBexp22,XGBtra2)
b3XGBtra = r2_score(XGBexp33,XGBtra3)
b4XGBtra = r2_score(XGBexp44,XGBtra4)
mse1XGBpre = mean_squared_error(XGBexp1,XGBpre1)
mse2XGBpre = mean_squared_error(XGBexp2,XGBpre2)
mse3XGBpre = mean_squared_error(XGBexp3,XGBpre3)
mse4XGBpre = mean_squared_error(XGBexp4,XGBpre4)
mse1XGBtra = mean_squared_error(XGBexp11,XGBtra1)
mse2XGBtra = mean_squared_error(XGBexp22,XGBtra2)
mse3XGBtra = mean_squared_error(XGBexp33,XGBtra3)
mse4XGBtra = mean_squared_error(XGBexp44,XGBtra4)
print(b1XGBpre,b2XGBpre,b3XGBpre,b4XGBpre)
print(b1XGBtra,b2XGBtra,b3XGBtra,b4XGBtra)
print(mse1XGBpre,mse2XGBpre,mse3XGBpre,mse4XGBpre)
print(mse1XGBtra,mse2XGBtra,mse3XGBtra,mse4XGBtra)