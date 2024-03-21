import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('insurance.csv')
y = df['charges']
df = df.drop('charges', axis=1)

le = LabelEncoder()
df['sex'] = le.fit_transform(df.sex.values)
df['smoker'] = le.fit_transform(df.smoker.values)

onehot_region = pd.get_dummies(df['region'])
df = df.drop('region', axis=1)
df = df.join(onehot_region.iloc[:, :-1])

df.insert(0, "intercept", 1)

NUM_TO_KEEP = 1000
df_train = df.iloc[:NUM_TO_KEEP, :]
y_train = y.iloc[:NUM_TO_KEEP]

df_test = df.iloc[NUM_TO_KEEP:, :]
y_test = y.iloc[NUM_TO_KEEP:]

with open('insurance_features.txt', 'w') as f:
    f.write(f'{df_train.shape[0]} {df_train.shape[1]}\n')

df_train.to_csv('insurance_features.txt', mode='a', sep=' ', header=False, index=False)
y_train.to_csv('insurance_target.txt', sep=' ', header=False, index=False)


with open('insurance_features_test.txt', 'w') as f:
    f.write(f'{df_test.shape[0]} {df_test.shape[1]}\n')

df_test.to_csv('insurance_features_test.txt', mode='a', sep=' ', header=False, index=False)
y_test.to_csv('insurance_target_test.txt', sep=' ', header=False, index=False)
