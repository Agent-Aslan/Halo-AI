#    Copyright 2018 D-Wave Systems Inc.

#    Licensed under the Apache License, Version 2.0 (the "License")
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http: // www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from __future__ import print_function, division

import sys

import numpy as np

from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.impute import SimpleImputer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from qboost import WeakClassifiers, QBoostClassifier, QboostPlus

from sklearn.model_selection import train_test_split



def metric(y, y_pred):

    return metrics.accuracy_score(y, y_pred)


def train_model(X_train, y_train, X_test, y_test, lmd):
    """
    Train qboost model

    :param X_train: train input
    :param y_train: train label
    :param X_test: test input
    :param y_test: test label
    :param lmd: lmbda to control regularization term
    :return:
    """
    NUM_READS = 3000
    NUM_WEAK_CLASSIFIERS = 35
    # lmd = 0.5
    TREE_DEPTH = 3

    # define sampler
    dwave_sampler = DWaveSampler(solver={'qpu': True})
    # sa_sampler = micro.dimod.SimulatedAnnealingSampler()
    emb_sampler = EmbeddingComposite(dwave_sampler)

    N_train = len(X_train)
    N_test = len(X_test)

    print("\n======================================")
    print("Train#: %d, Test: %d" %(N_train, N_test))
    print('Num weak classifiers:', NUM_WEAK_CLASSIFIERS)
    print('Tree depth:', TREE_DEPTH)


    # input: dataset X and labels y (in {+1, -1}

    # Preprocessing data
    # imputer = SimpleImputer()
    scaler = preprocessing.StandardScaler()     # standardize features
    normalizer = preprocessing.Normalizer()     # normalize samples

    # X = imputer.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_train = normalizer.fit_transform(X_train)

    # X_test = imputer.fit_transform(X_test)
    X_test = scaler.fit_transform(X_test)
    X_test = normalizer.fit_transform(X_test)

    ## Adaboost
    print('\nAdaboost')

    clf = AdaBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS)

    # scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print('fitting...')
    clf.fit(X_train, y_train)

    hypotheses_ada = clf.estimators_
    # clf.estimator_weights_ = np.random.uniform(0,1,size=NUM_WEAK_CLASSIFIERS)
    print('testing...')
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print('accu (train): %5.2f'%(metric(y_train, y_train_pred)))
    print('accu (test): %5.2f'%(metric(y_test, y_test_pred)))

    # Ensembles of Decision Tree
    print('\nDecision tree')

    clf2 = WeakClassifiers(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf2.fit(X_train, y_train)

    y_train_pred2 = clf2.predict(X_train)
    y_test_pred2 = clf2.predict(X_test)
    print(clf2.estimator_weights)

    print('accu (train): %5.2f' % (metric(y_train, y_train_pred2)))
    print('accu (test): %5.2f' % (metric(y_test, y_test_pred2)))

    # Ensembles of Decision Tree
    print('\nQBoost')

    DW_PARAMS = {'num_reads': NUM_READS,
                 'auto_scale': True,
                 # "answer_mode": "histogram",
                 'num_spin_reversal_transforms': 10,
                 # 'annealing_time': 10,
                 'postprocess': 'optimization',
                 }

    clf3 = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf3.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)

    y_train_dw = clf3.predict(X_train)
    y_test_dw = clf3.predict(X_test)

    print(clf3.estimator_weights)

    print('accu (train): %5.2f' % (metric(y_train, y_train_dw)))
    print('accu (test): %5.2f' % (metric(y_test, y_test_dw)))


    # Ensembles of Decision Tree
    print('\nQBoostPlus')
    clf4 = QboostPlus([clf, clf2, clf3])
    clf4.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
    y_train4 = clf4.predict(X_train)
    y_test4 = clf4.predict(X_test)
    print(clf4.estimator_weights)

    print('accu (train): %5.2f' % (metric(y_train, y_train4)))
    print('accu (test): %5.2f' % (metric(y_test, y_test4)))


    print("=============================================")
    print("Method \t Adaboost \t DecisionTree \t Qboost \t QboostIt")
    print("Train\t %5.2f \t\t %5.2f \t\t\t %5.2f \t\t %5.2f"% (metric(y_train, y_train_pred),
                                                               metric(y_train, y_train_pred2),
                                                               metric(y_train, y_train_dw),
                                                               metric(y_train, y_train4)))
    print("Test\t %5.2f \t\t %5.2f \t\t\t %5.2f \t\t %5.2f"% (metric(y_test, y_test_pred),
                                                              metric(y_test,y_test_pred2),
                                                              metric(y_test, y_test_dw),
                                                              metric(y_test, y_test4)))
    print("=============================================")

    # plt.subplot(211)
    # plt.bar(range(len(y_test)), y_test)
    # plt.subplot(212)
    # plt.bar(range(len(y_test)), y_test_dw)
    # plt.show()

    return

##########
#This section of code was added by Dr. Dani Caputi and Leo Madrid @ PEACE Inc.
    
#This is a pilot test of a basic machine learning algorithm for non-deterministic BMI

UseBytes = False # if True, uses integer bytes as training data instead of raw bits
BitsToSample = 20000 # samples this many bits (or bytes if UseBytes = True) before binary input selector

    
#Reads data file for testing and training and generates input matricies for the machine learning
readFile = open('C:/Users/Aslan/Documents/python-machine-learning/NED_Output/NED_1594413673058.txt', 'r')
sepfile = readFile.read().split('\n')
Rbyte=[]
feedback=[]
Rbits=[]
for a in range (0,len(sepfile)):
    if sepfile[a].startswith('Question'):
        di=0
        for b in range (a-5,a):
            if sepfile[b].startswith('Feedback'):
                di=1
        if di==0:
            for b in range (a-5,a):
                nodes = sepfile[b].split(',')
                for c in range (0,1000):
                    bt = int(nodes[c])
                    Rbyte.append(bt)
                    strbin = str(bin(256+bt)[3:])
                    for d in range (0,len(strbin)):
                        Rbits.append(int(strbin[d]))
        else:
            for b in range (a-6,a):
                nodes = sepfile[b].split(',')
                if len(nodes)>10:
                    for c in range (0,1000):
                        bt = int(nodes[c])
                        Rbyte.append(bt)
                        strbin = str(bin(256+bt)[3:])
                        for d in range (0,len(strbin)):
                            Rbits.append(int(strbin[d]))
    if sepfile[a].startswith('Feedback'):
        xandy = sepfile[a].split(',')
        feedback.append(str(xandy[1]))
    
    
    
    data=[]
    labels=[]
    
    for a in range (0,len(feedback)):
        if feedback[a]=='h' or feedback[a]=='m':
            if feedback[a]=='h':
                kval = 1
            else:
                kval = 0
            labels.append(kval)
            
            #ll=a*5000
            #ul=ll+5000
            nByte =[]
            if UseBytes==True:
                ll_offset = 5000-(BitsToSample+1)
                ll=(a*5000)+ll_offset#so that we don't include the last byte which contains the answer. on BYTE level seems to work well! but inconsistent and sometimes outputs that one way function problem.  try even less bytes/bits etc.  
                ul=ll+BitsToSample
                for b in range (ll,ul):
                    nByte.append((Rbyte[b]/255.0)-0.5)
            else:
                ll_offset = 40000-(BitsToSample+1)
                ll=(a*40000)+ll_offset#so that we don't include the last byte which contains the answer. on BYTE level seems to work well! but inconsistent and sometimes outputs that one way function problem.  try even less bytes/bits etc.  
                ul=ll+BitsToSample
                for b in range (ll,ul):
                    nByte.append((Rbits[b]))            
            
            data.append(nByte)
    
X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.5)

##########

clfs = train_model(X_train, y_train, X_test, y_test, 1.0)

