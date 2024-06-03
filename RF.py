import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
RFpre_new1 = pd.DataFrame()
RFexp_new1 = pd.DataFrame()
RFpre_new2 = pd.DataFrame()
RFexp_new2 = pd.DataFrame()
RFpre_new3 = pd.DataFrame()
RFexp_new3 = pd.DataFrame()
RFpre_new4 = pd.DataFrame()
RFexp_new4 = pd.DataFrame()
RFtra_new1 = pd.DataFrame()
RFexp_new11 = pd.DataFrame()
RFtra_new2 = pd.DataFrame()
RFexp_new22 = pd.DataFrame()
RFtra_new3 = pd.DataFrame()
RFexp_new33 = pd.DataFrame()
RFtra_new4 = pd.DataFrame()
RFexp_new44 = pd.DataFrame()
P1=[]
P2=[]
P3=[]
P4=[]
R1=[]
R2=[]
R3=[]
R4=[]
save_directory1 = "6regions-cross"
save_directory2 = "10regions-cross"
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
    for i in range(75,150):
        regressor = RandomForestRegressor(n_estimators=i + 1, n_jobs=-1)
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
    regressor1 = RandomForestRegressor(n_estimators=p1 + 76, n_jobs=-1)
    regressor2 = RandomForestRegressor(n_estimators=p2 + 76, n_jobs=-1)
    regressor3 = RandomForestRegressor(n_estimators=p3 + 76, n_jobs=-1)
    regressor4 = RandomForestRegressor(n_estimators=p4 + 76, n_jobs=-1)
    Hcmodel1 = regressor1.fit(X_cross1, Y_cross1)
    Hctrain1 = Hcmodel1.predict(X_cross1)
    Hctrain1 = pd.DataFrame(Hctrain1)
    Hcpre1 = Hcmodel1.predict(Xtest_1)
    Hcpre1 = pd.DataFrame(Hcpre1)
    RFpre_new1 = pd.concat([RFpre_new1, Hcpre1], axis=0, ignore_index=True)
    RFexp_new1 = pd.concat([RFexp_new1, Ytest_1], axis=0, ignore_index=True)
    RFtra_new1 = pd.concat([RFtra_new1, Hctrain1], axis=0, ignore_index=True)
    RFexp_new11 = pd.concat([RFexp_new11, Y_cross1], axis=0, ignore_index=True)
    Hcmodel2 = regressor2.fit(X_cross2, Y_cross2)
    Hctrain2 = Hcmodel2.predict(X_cross2)
    Hctrain2 = pd.DataFrame(Hctrain2)
    Hcpre2 = Hcmodel2.predict(Xtest_2)
    Hcpre2 = pd.DataFrame(Hcpre2)
    RFpre_new2 = pd.concat([RFpre_new2, Hcpre2], axis=0, ignore_index=True)
    RFexp_new2 = pd.concat([RFexp_new2, Ytest_2], axis=0, ignore_index=True)
    RFtra_new2 = pd.concat([RFtra_new2, Hctrain2], axis=0, ignore_index=True)
    RFexp_new22 = pd.concat([RFexp_new22, Y_cross2], axis=0, ignore_index=True)
    Hcmodel3 = regressor3.fit(X_cross3, Y_cross3)
    Hctrain3 = Hcmodel3.predict(X_cross3)
    Hctrain3 = pd.DataFrame(Hctrain3)
    Hcpre3 = Hcmodel3.predict(Xtest_3)
    Hcpre3 = pd.DataFrame(Hcpre3)
    RFpre_new3 = pd.concat([RFpre_new3, Hcpre3], axis=0, ignore_index=True)
    RFexp_new3 = pd.concat([RFexp_new3, Ytest_3], axis=0, ignore_index=True)
    RFtra_new3 = pd.concat([RFtra_new3, Hctrain3], axis=0, ignore_index=True)
    RFexp_new33 = pd.concat([RFexp_new33, Y_cross3], axis=0, ignore_index=True)
    Hcmodel4 = regressor4.fit(X_cross4, Y_cross4)
    Hctrain4 = Hcmodel4.predict(X_cross4)
    Hctrain4 = pd.DataFrame(Hctrain4)
    Hcpre4 = Hcmodel4.predict(Xtest_4)
    Hcpre4 = pd.DataFrame(Hcpre4)
    RFpre_new4 = pd.concat([RFpre_new4, Hcpre4], axis=0, ignore_index=True)
    RFexp_new4 = pd.concat([RFexp_new4, Ytest_4], axis=0, ignore_index=True)
    RFtra_new4 = pd.concat([RFtra_new4, Hctrain4], axis=0, ignore_index=True)
    RFexp_new44 = pd.concat([RFexp_new44, Y_cross4], axis=0, ignore_index=True)
with pd.ExcelWriter('RFpre_new.xlsx') as writer:
    pd.DataFrame(RFpre_new1).to_excel(writer, sheet_name='6', index=False)
    pd.DataFrame(RFpre_new2).to_excel(writer, sheet_name='10', index=False)
    pd.DataFrame(RFpre_new3).to_excel(writer, sheet_name='3', index=False)
    pd.DataFrame(RFpre_new4).to_excel(writer, sheet_name='12', index=False)
with pd.ExcelWriter('RFexp_new.xlsx') as writer2:
    pd.DataFrame(RFexp_new1).to_excel(writer2, sheet_name='6', index=False)
    pd.DataFrame(RFexp_new2).to_excel(writer2, sheet_name='10', index=False)
    pd.DataFrame(RFexp_new3).to_excel(writer2, sheet_name='3', index=False)
    pd.DataFrame(RFexp_new4).to_excel(writer2, sheet_name='12', index=False)
with pd.ExcelWriter('RFtra_new.xlsx') as writer2:
    pd.DataFrame(RFtra_new1).to_excel(writer2, sheet_name='6', index=False)
    pd.DataFrame(RFtra_new2).to_excel(writer2, sheet_name='10', index=False)
    pd.DataFrame(RFtra_new3).to_excel(writer2, sheet_name='3', index=False)
    pd.DataFrame(RFtra_new4).to_excel(writer2, sheet_name='12', index=False)
with pd.ExcelWriter('RFexp_new11.xlsx') as writer:
    pd.DataFrame(RFexp_new11).to_excel(writer, sheet_name='6', index=False)
    pd.DataFrame(RFexp_new22).to_excel(writer, sheet_name='10', index=False)
    pd.DataFrame(RFexp_new33).to_excel(writer, sheet_name='3', index=False)
    pd.DataFrame(RFexp_new44).to_excel(writer, sheet_name='12', index=False)
with pd.ExcelWriter('RFindex.xlsx') as index:
    pd.DataFrame(P1).to_excel(index, sheet_name='6', index=False)
    pd.DataFrame(P2).to_excel(index, sheet_name='10', index=False)
    pd.DataFrame(P3).to_excel(index, sheet_name='3', index=False)
    pd.DataFrame(P4).to_excel(index, sheet_name='12', index=False)
with pd.ExcelWriter('RFvalidR2.xlsx') as index:
    pd.DataFrame(R1).to_excel(index, sheet_name='6', index=False)
    pd.DataFrame(R2).to_excel(index, sheet_name='10', index=False)
    pd.DataFrame(R3).to_excel(index, sheet_name='3', index=False)
    pd.DataFrame(R4).to_excel(index, sheet_name='12', index=False)
RFpre1 = pd.read_excel('RFpre_new.xlsx',sheet_name='6')
RFpre2 = pd.read_excel('RFpre_new.xlsx',sheet_name='10')
RFpre3 = pd.read_excel('RFpre_new.xlsx',sheet_name='3')
RFpre4 = pd.read_excel('RFpre_new.xlsx',sheet_name='12')
RFexp1 = pd.read_excel('RFexp_new.xlsx',sheet_name='6')
RFexp2 = pd.read_excel('RFexp_new.xlsx',sheet_name='10')
RFexp3 = pd.read_excel('RFexp_new.xlsx',sheet_name='3')
RFexp4 = pd.read_excel('RFexp_new.xlsx',sheet_name='12')
RFtra1 = pd.read_excel('RFtra_new.xlsx',sheet_name='6')
RFtra2 = pd.read_excel('RFtra_new.xlsx',sheet_name='10')
RFtra3 = pd.read_excel('RFtra_new.xlsx',sheet_name='3')
RFtra4 = pd.read_excel('RFtra_new.xlsx',sheet_name='12')
RFexp11 = pd.read_excel('RFexp_new11.xlsx',sheet_name='6')
RFexp22 = pd.read_excel('RFexp_new11.xlsx',sheet_name='10')
RFexp33 = pd.read_excel('RFexp_new11.xlsx',sheet_name='3')
RFexp44 = pd.read_excel('RFexp_new11.xlsx',sheet_name='12')
b1RFpre = r2_score(RFexp1,RFpre1)
b2RFpre = r2_score(RFexp2,RFpre2)
b3RFpre = r2_score(RFexp3,RFpre3)
b4RFpre = r2_score(RFexp4,RFpre4)
b1RFtra = r2_score(RFexp11,RFtra1)
b2RFtra = r2_score(RFexp22,RFtra2)
b3RFtra = r2_score(RFexp33,RFtra3)
b4RFtra = r2_score(RFexp44,RFtra4)
mse1RFpre = mean_squared_error(RFexp1,RFpre1)
mse2RFpre = mean_squared_error(RFexp2,RFpre2)
mse3RFpre = mean_squared_error(RFexp3,RFpre3)
mse4RFpre = mean_squared_error(RFexp4,RFpre4)
mse1RFtra = mean_squared_error(RFexp11,RFtra1)
mse2RFtra = mean_squared_error(RFexp22,RFtra2)
mse3RFtra = mean_squared_error(RFexp33,RFtra3)
mse4RFtra = mean_squared_error(RFexp44,RFtra4)
print(b1RFpre,b2RFpre,b3RFpre,b4RFpre)
print(b1RFtra,b2RFtra,b3RFtra,b4RFtra)
print(mse1RFpre,mse2RFpre,mse3RFpre,mse4RFpre)
print(mse1RFtra,mse2RFtra,mse3RFtra,mse4RFtra)