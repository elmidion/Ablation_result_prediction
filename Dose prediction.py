
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
print(tf.__version__)
tf.random.set_seed(20200819)

import os
from time import time

Dose = pd.read_csv('/home/maitec/Projects/I-131 Dose Prediction/Dose prediction_20200819.csv')
#Dose.shape
#Dose.head

THW = Dose['preparation']==1
Dose_main = Dose[THW]

Dose_main.columns
Dose_main = Dose_main.drop(['ID', 'preadmissiondate', 'preparation', 'FUdate', 'FUscan', 'Furesult'], axis='columns')
#Dose_main.head
#Dose_main.describe()

Dose_main.pN[Dose_main.pN==' 1b'] = '1b'

# DATA Explore
'''
plt.hist(Dose_main.age, bins=10)
plt.show()

plt.hist(Dose_main.sex, bins=2)
plt.show()

plt.hist(Dose_main.biopsy, bins=2) # have to consider remove follicular type, because it is too small.
plt.show()

plt.hist(Dose_main.tumorsize) # Consider to apply log scale, because it shows right skewness
plt.show()

plt.hist(Dose_main.pT)
plt.show()

plt.hist(Dose_main.pN)
plt.show()

plt.hist(Dose_main.pM)
plt.show()

plt.hist(Dose_main.preTg) # have to apply log scale, because almost of preTg <10 
plt.show()

plt.hist(Dose_main.preATA) # have to apply log scale, because it has right skewness
plt.show()

plt.hist(Dose_main.preTSH)  
plt.show()

plt.hist(Dose_main.dose) 
plt.show()

plt.hist(Dose_main.preparation) 
plt.show()

plt.hist(Dose_main.Fubinary) 
plt.show()

'''
# DATA adjustment
#Dose_main.columns

# age MinMax scaling
Dose_main['age_MM'] = (Dose_main['age'] - min(Dose_main['age']))/(max(Dose_main['age']) - min(Dose_main['age']))

# Sex Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.sex)], axis=1)
Dose_main.rename(columns={1:'Male'}, inplace=True)
Dose_main.rename(columns={2:'Female'}, inplace=True)

# HTN Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.HTN)], axis=1)
Dose_main.rename(columns={0:'NoHTN'}, inplace=True)
Dose_main.rename(columns={1:'HTN'}, inplace=True)

# DM Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.DM)], axis=1)
Dose_main.rename(columns={0:'NoDM'}, inplace=True)
Dose_main.rename(columns={1:'DM'}, inplace=True)

# TB Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.TB)], axis=1)
Dose_main.rename(columns={0:'NoTB'}, inplace=True)
Dose_main.rename(columns={1:'TB'}, inplace=True)

# Hepatitis Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.Hepatitis)], axis=1)
Dose_main.rename(columns={0:'NoHepatitis'}, inplace=True)
Dose_main.rename(columns={1:'Hepatitis'}, inplace=True)

# OtherthyroidD Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.OtherthyroidD)], axis=1)
Dose_main.rename(columns={0:'NoOtherthyroidD'}, inplace=True)
Dose_main.rename(columns={1:'OtherthyroidD'}, inplace=True)

# Op Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.Op)], axis=1)

'''
# Biopsy Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.biopsy)], axis=1)
Dose_main.rename(columns={1:'Papillary'}, inplace=True)
Dose_main.rename(columns={2:'Follicular'}, inplace=True)
'''

# Tumorsize MinMax scaling
Dose_main['tumorsize_MM'] = (np.log(Dose_main['tumorsize']) - min(np.log(Dose_main['tumorsize'])))/(max(np.log(Dose_main['tumorsize'])) - min(np.log(Dose_main['tumorsize'])))

# pT Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.pT)], axis=1)
Dose_main.rename(columns={'1a':'pT1a'}, inplace=True)
Dose_main.rename(columns={'1b':'pT1b'}, inplace=True)
Dose_main.rename(columns={'2':'pT2'}, inplace=True)
Dose_main.rename(columns={'3a':'pT3a'}, inplace=True)
Dose_main.rename(columns={'4a':'pT4a'}, inplace=True)

# pN Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.pN)], axis=1)
Dose_main.rename(columns={'0':'N0'}, inplace=True)
Dose_main.rename(columns={'1a':'N1a'}, inplace=True)
Dose_main.rename(columns={'1b':'N1b'}, inplace=True)
Dose_main.rename(columns={' 1b':'N1b'}, inplace=True)

# pM Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.pM)], axis=1)
Dose_main.rename(columns={0:'M0'}, inplace=True)
Dose_main.rename(columns={1:'M1'}, inplace=True)

# cT Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.cT)], axis=1)
Dose_main.rename(columns={'1a':'cT1a'}, inplace=True)
Dose_main.rename(columns={'1b':'cT1b'}, inplace=True)
Dose_main.rename(columns={'2':'cT2'}, inplace=True)
Dose_main.rename(columns={'3a':'cT3a'}, inplace=True)
Dose_main.rename(columns={'3b':'cT3b'}, inplace=True)
Dose_main.rename(columns={'4a':'cT4a'}, inplace=True)


# preTg MinMax scaling
Dose_main['preTg_MM'] = (np.log(Dose_main['preTg']) - min(np.log(Dose_main['preTg'])))/(max(np.log(Dose_main['preTg'])) - min(np.log(Dose_main['preTg'])))

# preATA MinMax scaling
Dose_main['preATA_MM'] = (np.log(Dose_main['preATA']) - min(np.log(Dose_main['preATA'])))/(max(np.log(Dose_main['preATA']) )- min(np.log(Dose_main['preATA'])))

# preTSH MinMax scaling
Dose_main['preTSH_MM'] = (Dose_main['preTSH'] - min(Dose_main['preTSH']))/(max(Dose_main['preTSH']) - min(Dose_main['preTSH']))

'''
# preparation Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.preparation)], axis=1)
Dose_main.rename(columns={1:'THW'}, inplace=True)
Dose_main.rename(columns={2:'rhTSH'}, inplace=True)
'''

# Fubinary Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.Fubinary)], axis=1)
Dose_main.rename(columns={0:'AblationSuccess'}, inplace=True)
Dose_main.rename(columns={1:'AblationFailure'}, inplace=True)

'''
# dose Onehot encoding
Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.dose)], axis=1)
Dose_main.rename(columns={30:'30'}, inplace=True)
Dose_main.rename(columns={50:'50'}, inplace=True)
Dose_main.rename(columns={80:'80'}, inplace=True)
Dose_main.rename(columns={100:'100'}, inplace=True)
Dose_main.rename(columns={150:'150'}, inplace=True)
Dose_main.rename(columns={180:'180'}, inplace=True)
Dose_main.rename(columns={200:'200'}, inplace=True)
'''


# Delete original columns
Dose_main = Dose_main.drop(['age', 'sex', 'HTN', 'DM', 'TB', 'Hepatitis', 'OtherthyroidD', 'Op', 'biopsy', 'tumorsize', 'pT', 'pN', 'pM', 'cT', 'preTg', 'preATA', 'preTSH', 'Fubinary'], axis='columns')

#Dose_main.describe()
#Dose_main.columns

Dose_main_X = Dose_main[['dose', 'age_MM', 'Male', 'Female', 'NoHTN', 'NoDM', 'NoTB',
       'NoHepatitis', 'NoOtherthyroidD', 'TT', 'TT CND', 'TT RND',
       'tumorsize_MM', 'pT1a', 'pT1b', 'pT2', 'pT3a', 'pT4a', 'N0', 'N1a',
       'N1b', 'M0', 'M1', 'cT1a', 'cT1b', 'cT2', 'cT3a', 'cT3b', 'cT4a',
       'preTg_MM', 'preATA_MM', 'preTSH_MM',]]

Dose_main_Y = Dose_main[['AblationSuccess',
                         'AblationFailure']]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(Dose_main_X, Dose_main_Y, test_size= 0.2, shuffle=True, random_state=20200819)
X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy

# Create model
def create_model():
    model = Sequential()
    #model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.25))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.25))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 20200819
numpy.random.seed(seed)

# 모델 실행
model=create_model()
model.fit(X_train, y_train, epochs=1000, batch_size=X_train.shape[0])
#model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X_train, y_train, cv=kfold)

model.evaluate(X_test, y_test)
print(model.metrics_names)

yhat = model.predict(X_test)[:,1]
for idx in range(len(yhat)):
    if yhat[idx] >= 0.5:
        yhat[idx] = 1
    else:
        yhat[idx] = 0 

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test[:,1], yhat)



# # 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
# ## Accuracy is 0.8268, it is good.

# ###-------------------------------
# # Dose prediction using softmax

# #Dose_main.dose.value_counts()

# # dose Onehot encoding
# Dose_main = pd.concat([Dose_main, pd.get_dummies(Dose_main.dose)], axis=1)
# Dose_main.rename(columns={30:'30'}, inplace=True)
# Dose_main.rename(columns={50:'50'}, inplace=True)
# Dose_main.rename(columns={80:'80'}, inplace=True)
# Dose_main.rename(columns={100:'100'}, inplace=True)
# Dose_main.rename(columns={150:'150'}, inplace=True)
# Dose_main.rename(columns={180:'180'}, inplace=True)
# Dose_main.rename(columns={200:'200'}, inplace=True)
# Dose_main.rename(columns={300:'300'}, inplace=True)
# Dose_main.rename(columns={350:'350'}, inplace=True)

# Dose_main_X = Dose_main[['age_MM', 'Male', 'Female', 'NoHTN', 'NoDM', 'NoTB',
#        'NoHepatitis', 'NoOtherthyroidD', 'TT', 'TT CND', 'TT RND',
#        'tumorsize_MM', 'pT1a', 'pT1b', 'pT2', 'pT3a', 'pT4a', 'N0', 'N1a',
#        'N1b', 'M0', 'M1', 'cT1a', 'cT1b', 'cT2', 'cT3a', 'cT3b', 'cT4a',
#        'preTg_MM', 'preATA_MM', 'preTSH_MM','AblationSuccess', 'AblationFailure']]

# Dose_main_Y = Dose_main[['30', '50', '80', '100', '150', '180', '200', '300', '350']]

# # split dataset
# X_train, X_test, y_train, y_test = train_test_split(Dose_main_X, Dose_main_Y, test_size= 0.2, shuffle=True, random_state=20200819)
# X_train = np.asarray(X_train, dtype=np.float32)
# y_train = np.asarray(y_train, dtype=np.float32)
# X_test = np.asarray(X_test, dtype=np.float32)
# y_test = np.asarray(y_test, dtype=np.float32)

# from tensorflow import keras
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.models import Sequential
# #from sklearn.preprocessing import LabelEncoder
# #from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# #from keras.callbacks import ModelCheckpoint,EarlyStopping
# #from keras.wrappers.scikit_learn import KerasClassifier
# #from sklearn.model_selection import KFold
# #from sklearn.model_selection import cross_val_score
# #import numpy

# # Create model
# def create_model():
#     model = Sequential()
#     #model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     #model.add(Dropout(0.25))
#     model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dropout(0.25))
#     #model.add(Dense(16, activation='relu'))
#     #model.add(Dropout(0.25))
#     #model.add(Dense(8, activation='relu'))
#     #model.add(Dropout(0.25))
#     model.add(Dense(y_train.shape[1], activation='sigmoid'))
#     opt = keras.optimizers.Adam(learning_rate=0.01)
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     return model

# # fix random seed for reproducibility
# seed = 20200819
# numpy.random.seed(seed)

# # 모델 실행
# model=create_model()
# model.fit(X_train, y_train, epochs=1000, batch_size=X_train.shape[0])
# #model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# #kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# #results = cross_val_score(model, X_train, y_train, cv=kfold)


# # 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
#  ## Acurracy is 0.4409, too bad