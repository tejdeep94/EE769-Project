# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:23:17 2018

@author: god
"""


import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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
print('Calculating the mean vectors.....')

overall_mean = pd.DataFrame(columns=list(continuous_test_data))


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

overall_mean = overall_mean.append(continuous_test_data.mean(axis=0),ignore_index = True)


# Computing the scatter matrix #
print('Computing the scatter matrix.....')
scatter_matrix = np.zeros((len(continuous_test_data.columns),len(continuous_test_data.columns)))
for i in range(0,len(continuous_test_data)):
    temp = (continuous_test_data.loc[i]-overall_mean)
    temp1 = np.matmul((temp.T).as_matrix(),temp.as_matrix())
    scatter_matrix = scatter_matrix + temp1


val,vec = np.linalg.eig(scatter_matrix)
eig_pairs = [(np.abs(val[i]), vec[:,i]) for i in range(len(val))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

new_feature = np.zeros((len(vec),60))

for i in range(0,60):
    new_feature[:,i] = (eig_pairs[i][1])
transformed_samples = np.matmul(continuous_test_data.as_matrix(),new_feature)   
transformed_samples = preprocessing.normalize(transformed_samples, norm='l2')

#---------------------------------- CLASSIFIERS ------------------------------#
# Neural Network Classifier #
NN_model = pickle.load(open('PCA_NN_model.sav', 'rb'))
predicted_label = NN_model.predict(transformed_samples)
print('The test accuracy score for NN classifier is : ')
print (accuracy_score(y, predicted_label))  

# Decision tree Classifier #
Tree_model = pickle.load(open('PCA_Tree_model.sav', 'rb'))
predicted_label = Tree_model.predict(transformed_samples)
print('The test accuracy score for Tree classifier is : ')
print (accuracy_score(y, predicted_label))