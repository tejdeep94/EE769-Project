# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:55:21 2018

@author: god
"""


import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

print('Loading data ......')
test_data = pd.read_csv('KDDTest+.csv')
temp_x, waste_x = train_test_split(test_data, test_size=0.05, random_state=42)
test_data = temp_x

test_data = test_data.dropna(axis=0, how='any')
test_data = test_data.reset_index()

categories = pd.read_csv('Field Names.csv')
label = test_data['label']
difficulty = test_data['difficulty']
test_data = test_data.drop(['index','label','difficulty'],axis = 1)

# Separating the symbolic data from the continous data #
print('Preprocessing the data.....')
symbolic_test_data = test_data[['protocol_type','service','flag']]
#continuous_test_data = test_data.drop(['protocol_type','service','flag'],axis = 1)
continuous_test_data = pd.get_dummies(test_data)
store = list(continuous_test_data)
continuous_test_data = preprocessing.scale(continuous_test_data)
continuous_test_data = preprocessing.normalize(continuous_test_data, norm='l2')
continuous_test_data = pd.DataFrame(continuous_test_data,columns=store)

# Calculating the feature mean vector of every class #
print('Calculating the mean vectors for PCA.....')

overall_mean_PCA = pd.DataFrame(columns=list(continuous_test_data))


# Assigning labels to categories #

for i in range(0,len(label)):
    print(i)
    if label.loc[i] == 'back' or label.loc[i] == 'land' or label.loc[i] == 'mailbomb' or label.loc[i] == 'neptune' or label.loc[i] == 'pod' or label.loc[i] == 'smurf' or label.loc[i] == 'teardrop' or label.loc[i] == 'back' or label.loc[i] == 'apache2' or label.loc[i] == 'udpstorm' or label.loc[i] == 'processtable' or label.loc[i] == 'worm':
        label.loc[i] = 'DOS'
        
    elif label.loc[i] == 'buffer_overflow' or label.loc[i] == 'loadmodule' or label.loc[i] == 'perl' or label.loc[i] == 'rootkit' or label.loc[i] == 'sqlattack' or label.loc[i] == 'xterm' or label.loc[i] == 'ps  ':
        label.loc[i] = 'U2R'
        
    elif label.loc[i] == 'guess_passwd' or label.loc[i] == 'ftp_write' or label.loc[i] == 'ps' or label.loc[i] == 'imap' or label.loc[i] == 'multihop' or label.loc[i] == 'phf' or label.loc[i] == 'spy' or label.loc[i] == 'warezclient' or label.loc[i] == 'warezmaster'or label.loc[i] == 'xlock' or label.loc[i] == 'xsnoop' or label.loc[i] == 'snmpguess' or label.loc[i] == 'snmpgetattack' or label.loc[i] == 'httptunnel'or label.loc[i] == 'sendmail' or label.loc[i] == 'named':
        label.loc[i] = 'R2L'
        
    elif label.loc[i] == 'ipsweep' or label.loc[i] == 'nmap' or label.loc[i] == 'portsweep' or label.loc[i] == 'satan'or label.loc[i] == 'saint' or label.loc[i] == 'mscan':
        label.loc[i] = 'PROBE'
        

        
# Assigning numerical labels to the class of attacks #

class_mapping = {'normal' : 0,'DOS': 1,'U2R' : 2,'R2L' : 3,'PROBE' : 4}
y = label.map(class_mapping)


# Computing the mean vectors #

overall_mean_PCA = overall_mean_PCA.append(continuous_test_data.mean(axis=0),ignore_index = True)


# Computing the scatter matrix for PCA #
print('Computing the scatter matrix for PCA.....')
scatter_matrix_PCA = np.zeros((len(continuous_test_data.columns),len(continuous_test_data.columns)))
for i in range(0,len(continuous_test_data)):
    temp = (continuous_test_data.loc[i]-overall_mean_PCA)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    scatter_matrix_PCA = scatter_matrix_PCA + temp1


val_PCA,vec_PCA = np.linalg.eig(scatter_matrix_PCA)
eig_pairs_PCA = [(np.abs(val_PCA[i]), vec_PCA[:,i]) for i in range(len(val_PCA))]
eig_pairs_PCA = sorted(eig_pairs_PCA, key=lambda k: k[0], reverse=True)

new_feature = np.zeros((len(vec_PCA),60))

for i in range(0,60):
    new_feature[:,i] = (eig_pairs_PCA[i][1])
transformed_samples = np.matmul(continuous_test_data.as_matrix(),new_feature)   
transformed_samples = preprocessing.normalize(transformed_samples, norm='l2')

transformed_samples = pd.DataFrame(transformed_samples)

# Calculating the feature mean vector of every class #
print('Calculating the mean vectors for each class for LDA .....')
normal_data = pd.DataFrame(columns=list(transformed_samples))
print('moving forward')
normal_mean = pd.DataFrame(columns=list(transformed_samples))
DOS_data = pd.DataFrame(columns=list(transformed_samples))
DOS_mean = pd.DataFrame(columns=list(transformed_samples))
U2R_data = pd.DataFrame(columns=list(transformed_samples))
U2R_mean = pd.DataFrame(columns=list(transformed_samples))
R2L_data = pd.DataFrame(columns=list(transformed_samples))
R2L_mean = pd.DataFrame(columns=list(transformed_samples))
PROBE_data = pd.DataFrame(columns=list(transformed_samples))
PROBE_mean = pd.DataFrame(columns=list(transformed_samples))
print('moving forward')
# Assigning labels to categories #

for i in range(0,len(label)):
    print(i)
    if label.loc[i] == 'DOS':
        DOS_data = DOS_data.append(transformed_samples.loc[i])
    elif label.loc[i] == 'U2R':
        U2R_data = U2R_data.append(transformed_samples.loc[i])
    elif label.loc[i] == 'R2L':
        R2L_data = R2L_data.append(transformed_samples.loc[i])
    elif label.loc[i] == 'PROBE':
        PROBE_data = PROBE_data.append(transformed_samples.loc[i])
    elif label.loc[i] == 'normal':
       normal_data = normal_data.append(transformed_samples.loc[i])

normal_data = normal_data.reset_index()
normal_data = normal_data.drop(['index'],axis=1)
DOS_data = DOS_data.reset_index()
DOS_data = DOS_data.drop(['index'],axis=1)
U2R_data = U2R_data.reset_index()
U2R_data = U2R_data.drop(['index'],axis=1)
R2L_data = R2L_data.reset_index()
R2L_data = R2L_data.drop(['index'],axis=1)
PROBE_data = PROBE_data.reset_index()
PROBE_data = PROBE_data.drop(['index'],axis=1)

# Computing the mean vectors of each class #

normal_mean = normal_mean.append(normal_data.mean(axis=0),ignore_index = True)
DOS_mean = DOS_mean.append(DOS_data.mean(axis=0),ignore_index = True)
U2R_mean = U2R_mean.append(U2R_data.mean(axis=0),ignore_index = True)
R2L_mean = R2L_mean.append(R2L_data.mean(axis=0),ignore_index = True)
PROBE_mean = PROBE_mean.append(PROBE_data.mean(axis=0),ignore_index = True)

# Computing the scatter matrices #
print('Computing the scatter matrices for LDA.....')
normal_scatter = np.zeros((len(normal_data.columns),len(normal_data.columns)))
for i in range(0,len(normal_data)):
    temp = (normal_data.loc[i]-normal_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    normal_scatter = normal_scatter + temp1

DOS_scatter = np.zeros((len(DOS_data.columns),len(DOS_data.columns)))
for i in range(0,len(DOS_data)):
    temp = (DOS_data.loc[i]-DOS_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    DOS_scatter = DOS_scatter + temp1
    
U2R_scatter = np.zeros((len(U2R_data.columns),len(U2R_data.columns)))
for i in range(0,len(U2R_data)):
    temp = (U2R_data.loc[i]-U2R_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    U2R_scatter = U2R_scatter + temp1
    
R2L_scatter = np.zeros((len(R2L_data.columns),len(R2L_data.columns)))
for i in range(0,len(R2L_data)):
    temp = (R2L_data.loc[i]-R2L_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    R2L_scatter = R2L_scatter + temp1
    
PROBE_scatter = np.zeros((len(PROBE_data.columns),len(PROBE_data.columns)))
for i in range(0,len(PROBE_data)):
    temp = (PROBE_data.loc[i]-PROBE_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    PROBE_scatter = PROBE_scatter + temp1

scatter_matrix_LDA = normal_scatter + DOS_scatter + U2R_scatter + R2L_scatter + PROBE_scatter

# Computing the overall scatter matrix #
print('Computing the overall scatter matrix for LDA.....')
overall_mean_LDA = pd.DataFrame(columns=list(transformed_samples))
overall_mean_LDA = overall_mean_LDA.append(transformed_samples.mean(axis=0),ignore_index = True)

normal_temp = np.matmul(((normal_mean-overall_mean_LDA).T).as_matrix(),(normal_mean-overall_mean_LDA).as_matrix())
DOS_temp = np.matmul(((DOS_mean-overall_mean_LDA).T).as_matrix(),(DOS_mean-overall_mean_LDA).as_matrix())
U2R_temp = np.matmul(((U2R_mean-overall_mean_LDA).T).as_matrix(),(U2R_mean-overall_mean_LDA).as_matrix())
R2L_temp = np.matmul(((R2L_mean-overall_mean_LDA).T).as_matrix(),(R2L_mean-overall_mean_LDA).as_matrix())
PROBE_temp = np.matmul(((PROBE_mean-overall_mean_LDA).T).as_matrix(),(PROBE_mean-overall_mean_LDA).as_matrix())

overall_scatter_matrix_LDA =len(normal_data)*normal_temp + len(DOS_data)*DOS_temp + len(U2R_data)*U2R_temp + len(R2L_data)*R2L_temp + len(PROBE_data)*PROBE_temp

temp = np.linalg.lstsq(scatter_matrix_LDA,overall_scatter_matrix_LDA)
val_LDA,vec_LDA = np.linalg.eig(temp[0])
eig_pairs_LDA = [(np.abs(val_LDA[i]), vec_LDA[:,i]) for i in range(len(val_LDA))]
eig_pairs_LDA = sorted(eig_pairs_LDA, key=lambda k: k[0], reverse=True)

new_feature_final = np.zeros((len(vec_LDA),10))
for i in range(0,10):
    new_feature_final[:,i] = (eig_pairs_LDA[i][1])
transformed_samples_final = np.matmul(transformed_samples.as_matrix(),new_feature_final)   
transformed_samples_final = preprocessing.normalize(transformed_samples_final, norm='l2')

#---------------------------------- CLASSIFIERS ------------------------------#
# Neural Network Classifier #
NN_model = pickle.load(open('PCA_LDA_NN_model.sav', 'rb'))
predicted_label = NN_model.predict(transformed_samples_final)
print('The test accuracy score for NN classifier is : ')
print (accuracy_score(y, predicted_label)) 

# Decision tree Classifier #
Tree_model = pickle.load(open('PCA_LDA_Tree_model.sav', 'rb'))
predicted_label = Tree_model.predict(transformed_samples_final)
print('The test accuracy score for Tree classifier is : ')
print (accuracy_score(y, predicted_label))

