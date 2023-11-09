import pickle

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



df = pd.read_excel('Pumpkin_Seeds_Dataset/Pumpkin_Seeds_Dataset.xlsx')


# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# Class Mapping of seed name column for training
class_mapping = {'Çerçevelik': 0, 'Ürgüp Sivrisi': 1}
df['Class'] = df['Class'].map(class_mapping)


df.columns = df.columns.str.lower().str.replace(' ', '_')


# Feature Elimination: dropping perimeter, major_axis_length, convex_area, equiv_diameter, aspect_ratio, and compactness to reduce model complexity. These features were dropped due to high correlation with other features determined by FixDQ (identified high correlation features: area, eccentricity, roundness, convex_area)

X = df.drop(columns=['class'])#, 'perimeter', 'convex_area', 'equiv_diameter'])#, 'aspect_ration', 'compactness'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
std  = np.sqrt(scaler.var_)
np.save('std.npy',std)
np.save('mean.npy',scaler.mean_)

# training 

def train(X, y, C=10):
    model = SVC(C = C, probability=True)
    model.fit(X, y)
    return model

def predict(df, model):
    y_pred = model.predict(df)
    return y_pred



# validation

print(f'doing validation with C={C}')

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []

def kfold_fold_score(C=1.0):
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = train(X_train, y_train, C=C)
        y_pred = predict(X_test, model)
    
        auc = roc_auc_score(y_test, y_pred)
        scores.append(auc)
        print(f'auc on fold {i+1} is {auc}')


    print('validation results:')
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# training the final model

print('training the final model')

model = train(X, y, C=1.0)
y_pred = predict(X, model)
auc = roc_auc_score(y, y_pred)

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f'the model is saved to {output_file}')