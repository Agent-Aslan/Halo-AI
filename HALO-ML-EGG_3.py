# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:01:57 2020

@author: Aslan
"""

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

from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import csv
from pylab import polyfit

from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.impute import SimpleImputer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from qboost import WeakClassifiers, QBoostClassifier, QboostPlus

from sklearn.model_selection import train_test_split

import time


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

#mo = 7
#dy = 16

#strday = 'ttt%dttt%d'%(mo,dy)

#for day in range (1,17):

#GCPdat = 'http://global-mind.org/cgi-bin/eggdatareq.pl?z=1&year=2020&month=%s&day=%s&stime=00%3A00%3A00&etime=23%3A59%3A59&gzip=No&idate=Yes'%(mo,dy)
#..

"""
1 Princeton, NJ, USA        #
37 Neuchatel, Switzerland
110 Black Forest, CO, USA   #
112 Neuchatel, Switzerland
226 London, Ontario, CA
228 Hessdalen, Norway
1021 Santa Clara, CA, USA   ##
1070 Cincinnati, Ohio, USA  #
1092 Helsinki, Finland  
1237 Moscow, Russia
2000 Oconomowoc, WI, USA    #
2028 Soest, The Netherlands
2052 Athens, Greece
2080 Santiago, Chile
2083 Hsin Chu City, Taiwan
2178 Boulder Creek, CA, USA ##
2220 Gibsonburg, Ohio, USA  #
2232 Perth, Western Australia
2250 Ottawa, Canada
3066 Madrid, Spain
3101 Boulder Creek, CA, USA ##
3104 Lewes, East Susses, UK
3106 Coronation, Alberta, CA
3247 West Fork, AR, USA     #
4234 Seoul, Korea
"""




def GetEgg(text,egg,Fill=True):
    eggs = text[8].split(',')
    found=-999
    for a in range (3,len(eggs)):
        if egg==int(eggs[a]):
            found = a
    eggseq=[]
    if found<0 and Fill==True:
        eggseq = np.random.randint(0,2,86400)
        outfile.write('%s'%(eggseq))
    else:
        for a in range (9,len(text)-1):
            sp = text[a].split(',')
            if sp[found] != "":
                eggseq.append(int(sp[found]))
            else:
                eggseq.append(np.random.randint(0,2))#replace missing with pseudo-random number
            #
    return eggseq

#starttime = int(time.time()*1000)
outfile = open('C:/Users/Aslan/Documents/HALO Development/NED_Data/EGGML_10.txt','w')

COVdat = 'http://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv'
uClient = uReq(COVdat)
page_csv = str(uClient.read())
uClient.close()

sepfile = page_csv.split('\\r\\n')

"""
for a in range (0,len(sepfile)):
    if 'Santa Cruz' in sepfile[a]:
        print(a,"cruz")
    if 'Santa Clara' in sepfile[a]:
        print(a,"clara")

#236 is santa clara, 237 is santa cruz
jday1 = 22, start on 23 though with new cases per day
"""

xc = sepfile[236].split(',')[4:]
SantaClaraCases=[]
for a in range (1,len(xc)):
    SantaClaraCases.append(int(xc[a])-int(xc[a-1]))

xc = sepfile[237].split(',')[4:]
SantaCruzCases=[]
for a in range (1,len(xc)):
    SantaCruzCases.append(int(xc[a])-int(xc[a-1]))
    

    
cases = np.array(SantaCruzCases)+np.array(SantaClaraCases)
jday = np.arange(23,len(cases)+23)#start on 23rd day of year.

"""
TestDay = [69,99,130,160]
Tests = [32944,685048,2071591,4338718]

#approximate tests per day, CA
R = polyfit(TestDay,Tests,2)
TestCA = (2*R[0]*np.array(jday))+R[1]
"""

AMP=[]
for a in range (0,len(jday)):
    if jday[a]<=160:
        AMP.append(((-1/80)*jday[a])+3)
    else:
        AMP.append(1)
        
CasesAmp = np.array(cases)*AMP

SmoothCasesAmp=[]
SmoothJday=[]
for a in range (3,len(CasesAmp)-3):
    SmoothCasesAmp.append(np.average(CasesAmp[a-3:a+4]))
    SmoothJday.append(jday[a])

#cr = csv.reader(uClient)

#al = []
#for row in cr:
#    al.append(row)


#might need to do some smoothing of all days
#consider previous testing difficulty



months = [3,4,5,6,7,8]
DIM = [31,30,31,30,31,10]
    


Stream1=[]
Stream2=[]
Stream3=[]

for mm in range (0,len(months)):


    for dy in range (1,DIM[mm]+1):
    
        GCPdat = 'http://global-mind.org/cgi-bin/eggdatareq.pl?z=1&year=2020&month=' + str(months[mm]) + '&day=' + str(dy) + '&stime=00%3A00%3A00&etime=23%3A59%3A59&gzip=No&idate=Yes'
        
        uClient = uReq(GCPdat)
        page_html = str(uClient.read())
        uClient.close()
        
            
        sepfile = page_html.split('\\n')
        #print('%d/%d: %s'%(months[mm],dy,sepfile[8][24:]))
        
        
        s1 = GetEgg(sepfile,1021)
        s2 = GetEgg(sepfile,2178)
        s3 = GetEgg(sepfile,3101)

        Stream1.append(s1)
        Stream2.append(s2)
        Stream3.append(s3)
        
        
        if (len(s1)+len(s2)+len(s3)==259200):
            sstr = "ok"
        else:
            sstr = "BAD GCP"
        print('%d/%d %s %d %d %d'%(months[mm],dy,sstr,len(s1),len(s2),len(s3)))

data = []
labels = []

#start = SmoothJday[38]. 

MedianCOV = np.median(SmoothCasesAmp[38:])

CaseIdx = 38
for a in range (3,len(Stream1)-5):
    
    if (SmoothCasesAmp[CaseIdx]>MedianCOV):
        labels.append(1)
    else:
        labels.append(0)
    CaseIdx += 1
    
    matchstream = []
    for b in range(a-3,a):
        s1 = Stream1[b]
        s2 = Stream2[b]
        s3 = Stream3[b]
        for c in range (0,86400):
            if s1[c]==s2[c]==s3[c]:
                matchstream.append(1)
            else:
                matchstream.append(0)
    data.append(matchstream)
    
X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.5)

##########

clfs = train_model(X_train, y_train, X_test, y_test, 1.0)


