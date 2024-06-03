import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

df1 = pd.read_excel('solubility-types.xlsx',sheet_name='Sheet1')
df2 = pd.read_excel('solubility-types.xlsx',sheet_name='Sheet3')
df3 = pd.read_excel('solubility-types.xlsx',sheet_name='Sheet4')
df4 = pd.read_excel('solubility-types.xlsx',sheet_name='Sheet5')
X1=df1.iloc[:,2:10]
X2=df2.iloc[:,2:14]
X3=df3.iloc[:,2:7]
X4=df4.iloc[:,2:16]
X1 = pd.DataFrame(X1)
X2 = pd.DataFrame(X2)
X3 = pd.DataFrame(X3)
X4 = pd.DataFrame(X4)
Y = pd.DataFrame(df1.iloc[:, -1])
type = df1.iloc[:, 1]
data = pd.concat([type, X1, Y], axis=1)
unique_names = list(set(type))
unique_names_stand = list(set(type))


num_folds = 106
kfold = KFold(n_splits=num_folds, shuffle=True)

accuracy_scores = []

save_directory1 = "6regions-cross"
save_directory2 = "10regions-cross"
save_directory3 = "3regions-cross"
save_directory4 = "12regions-cross"


for fold in range(1,11):
    if len(unique_names) > 10:
        test_names = np.random.choice(unique_names, size=11, replace=False)
        unique_names[:] = [value for value in unique_names if value not in test_names]
        print(test_names)
    
        train_names = np.setdiff1d(unique_names_stand, np.concatenate([test_names]))

        train_data = df1[data['Type'].isin(train_names)]
       
        test_data = df1[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory1}fold_{fold}_train.csv", index=False)
       
        test_data.to_csv(f"{save_directory1}fold_{fold}_test.csv", index=False)

        train_data = df2[data['Type'].isin(train_names)]
       
        test_data = df2[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory2}fold_{fold}_train.csv", index=False)
       
        test_data.to_csv(f"{save_directory2}fold_{fold}_test.csv", index=False)

        train_data = df3[data['Type'].isin(train_names)]
       
        test_data = df3[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory3}fold_{fold}_train.csv", index=False)
       
        test_data.to_csv(f"{save_directory3}fold_{fold}_test.csv", index=False)

        train_data = df4[data['Type'].isin(train_names)]
       
        test_data = df4[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory4}fold_{fold}_train.csv", index=False)
      
        test_data.to_csv(f"{save_directory4}fold_{fold}_test.csv", index=False)
    else:

        test_names =unique_names  
        print(test_names)
        train_names = np.setdiff1d(unique_names_stand, np.concatenate([test_names]))

        train_data = df1[data['Type'].isin(train_names)]
       
        test_data = df1[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory1}fold_{fold}_train.csv", index=False)
       
        test_data.to_csv(f"{save_directory1}fold_{fold}_test.csv", index=False)

        train_data = df2[data['Type'].isin(train_names)]
       
        test_data = df2[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory2}fold_{fold}_train.csv", index=False)
        
        test_data.to_csv(f"{save_directory2}fold_{fold}_test.csv", index=False)

        train_data = df3[data['Type'].isin(train_names)]
     
        test_data = df3[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory3}fold_{fold}_train.csv", index=False)
        
        test_data.to_csv(f"{save_directory3}fold_{fold}_test.csv", index=False)

        train_data = df4[data['Type'].isin(train_names)]
        
        test_data = df4[data['Type'].isin(test_names)]

        train_data.to_csv(f"{save_directory4}fold_{fold}_train.csv", index=False)
       
        test_data.to_csv(f"{save_directory4}fold_{fold}_test.csv", index=False)

