#Experimental calculation for preprocessed neuroimages

# Enter input folder path - line 61
# Specify the method to perform experimental analysis - line 96 to 100
from scipy.io import loadmat

import numpy as np
from scipy import sparse
import pandas as pd
import glob
import os

import networkx as nx

from sknetwork.utils import edgelist2adjacency, edgelist2biadjacency
from sknetwork.data import convert_edge_list, load_edge_list, load_graphml
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
from sknetwork.clustering import Louvain, modularity, bimodularity
from sknetwork.ranking import Betweenness, Closeness

from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.metrics import recall_score, precision_score, accuracy_score, make_scorer
from sklearn import preprocessing

"""##### parameters"""

dataset = ['adni', 'abide', 'taowu', 'neurocon', 'ppmi'] #options are taowu, neurocon, abide, adni, ppmi
#dataset = ['taowu'] #options are taowu, neurocon, abide, adni, ppmi
parcellation = ['harvard48', 'AAL116', 'schaefer100', 'kmeans100', 'ward100'] #options are AAL116, harvard48, schaefer100, kmeans100, ward100
#parcellation = ['kmeans100'] #options are AAL116, harvard48, schaefer100, kmeans100, ward100
random_seed = 1
splits = 5

"""##### assign roi counts to specified parcellation schemes"""

num_roi = []
for x in range(0,len(parcellation)):
    if parcellation[x] == 'AAL116':
        num_roi.append(116)
    elif parcellation[x] == 'harvard48':
        num_roi.append(48)
    elif parcellation[x] == 'schaefer100':
        num_roi.append(100)
    elif parcellation[x] == 'kmeans100':
        num_roi.append(100)
    elif parcellation[x] == 'ward100':
        num_roi.append(100)

print(num_roi)

"""##### load data, create feature matrix by loading edge weights into X, and load class labels into y"""

def load_data(dataset, parcellation, num_rois):
    subject_paths = glob.glob('datasets/' + dataset + '/*')
    subject_names = [os.path.basename(x) for x in subject_paths]
    y = []
    X = np.zeros(num_rois*num_rois)
    
    for x in range(0,len(subject_paths)):
        mat = loadmat(subject_paths[x] + '/' + subject_names[x] + '_' + parcellation + '_correlation_matrix.mat')
        adjacency = mat['data']
    
        X = np.vstack([X,adjacency.flatten()])
    
        #generate class label list y based on subject ID
        if 'control' in subject_names[x]:
            y.append(1)
        elif 'patient' in subject_names[x]:
            y.append(2)
        elif 'mci' or 'prodromal' in subject_names[x]:
            y.append(3)
        elif 'emci' or 'swedd' in subject_names[x]:
            y.append(4)
        elif 'SMC' in subject_names[x]:
            y.append(5)
        elif 'LMCI' in subject_names[x]:
            y.append(6)
        
    X = np.delete(X, 0, axis=0) #delete empty first row of zeros from X

    return [X, y]

"""##### cross-validation"""

def cross_validation(splits, X, y, dataset, parcellation):
    cv_all = np.zeros(splits)

    for x in range(splits):
        clf_all = LogisticRegression(max_iter=1000000)
#        clf_all = KNeighborsClassifier(n_neighbors=5)
#        clf_all = SVC(gamma='auto')
#        clf_all = GaussianNB()
#        clf_all = RandomForestClassifier(max_depth=2, random_state=0)
        scoring = {'accuracy': make_scorer(accuracy_score)}
        cv = ShuffleSplit(n_splits=splits, test_size=1/splits, random_state=x*random_seed)
        scores_all = cross_validate(clf_all, X, y, cv=cv,scoring=scoring)
        cv_all = np.vstack([cv_all, scores_all['test_accuracy']])

    cv_all = np.delete(cv_all, 0, axis=0)
    print('------------------------------')
    print('dataset:',dataset)
    print('parcellation:',parcellation)
    print('random_seed:',random_seed)
    print('splits:',splits)
    print('------------------------------')
    print('Using all edges:')
    print('accuracy =',round(np.sum(cv_all)/(len(cv_all)*splits),3),'\u00B1',round(np.std(cv_all),3))
    print('------------------------------')

"""##### call methods to run experiments"""

for i in range(0,len(dataset)):
    for j in range(0,len(parcellation)):
        data = load_data(dataset[i], parcellation[j], num_roi[j])
        cross_validation(splits, data[0], data[1], dataset[i], parcellation[j])

