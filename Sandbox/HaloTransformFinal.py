# -*- coding: utf-8 -*-
"""Created on Sat Dec 19 22:47:20 2020.

@author: Danny
"""

#Make sure numpy is imported
import numpy as np
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OX_REGIONS_FILE = os.path.join(ROOT_DIR, "text_files", "OxRegions2.txt")
WORDS_FILE = os.path.join(ROOT_DIR, "text_files", "words_219k_s3514.txt")
COVMODEL_FILE = os.path.join(ROOT_DIR, "text_files", "CovModel_modfile_1608584471.txt")


#set parameters
SF = 1# standard errors get multiplied by this amount to weigh this HALO transformation
LenOutfile = 100# how many top words from Language Oracle to write to new file

#backend stuff, just make sure you have access to the files pulled for reading

ISPctry=[]
ISPstate=[]
SPQ = []
readFile = open(OX_REGIONS_FILE,encoding='latin-1')
sepfile = readFile.read().split('\n')
for a in range (0,len(sepfile)):
    xandy = sepfile[a].split(',')
    ISPctry.append(xandy[0])
    ISPstate.append(xandy[1])
    SPQ.append(int(xandy[2]))

halo_p=[]
halo_country=[]
halo_region=[]

LOnum=[]
readFile = open(COVMODEL_FILE,encoding='latin-1')
sepfile = readFile.read().split('\n')
for a in range (0,len(sepfile)-1):
    xandy = sepfile[a].split(',')
    for b in range (0,75000):
        LOnum.append(int(xandy[b]))
    halo_country.append(xandy[-10])
    halo_region.append(xandy[-9])
    halo_p.append(xandy[-8:])
    
intlseed = np.random.randint(0,16000000)
    
Blog=[]
Web=[]
TV=[]
Spoken=[]
Fiction=[]
Magazine=[]
News=[]
Academic=[]

Word=[]
Readfile=open(WORDS_FILE,encoding='latin-1')
Lines=Readfile.read().split('\n')
for line in range(9,len(Lines)-1):
    items=Lines[line].split('\t')
    Word.append(items[1])
    Blog.append(int(items[5]))
    Web.append(int(items[6]))
    TV.append(int(items[7]))
    Spoken.append(int(items[8]))
    Fiction.append(int(items[9]))
    Magazine.append(int(items[10]))
    News.append(int(items[11]))
    Academic.append(int(items[12]))

sBlogWord=[x for _,x in sorted (zip(Blog,Word),reverse=True)]
sWebWord=[x for _,x in sorted (zip(Web,Word),reverse=True)]
sTVWord=[x for _,x in sorted (zip(TV,Word),reverse=True)]
sSpokenWord=[x for _,x in sorted (zip(Spoken,Word),reverse=True)]
sFictionWord=[x for _,x in sorted (zip(Fiction,Word),reverse=True)]
sMagazineWord=[x for _,x in sorted (zip(Magazine,Word),reverse=True)]
sNewsWord=[x for _,x in sorted (zip(News,Word),reverse=True)]
sAcademicWord=[x for _,x in sorted (zip(Academic,Word),reverse=True)]

OutputLength=len(Word)

#2^17 total length

C_Blog=sBlogWord[0:131072]
C_Web=sWebWord[0:131072]
C_TV=sTVWord[0:131072]
C_Spoken=sSpokenWord[0:131072]
C_Fiction=sFictionWord[0:131072]
C_Magazine=sMagazineWord[0:131072]
C_News=sNewsWord[0:131072]
C_Academic=sAcademicWord[0:131072]

DIM = [31,28,31,30,31,30,31,31,30,31,30,31]
HALOdy = np.arange(0,200,26)
HALOdyall = np.arange(0,500)

def HaloTransform(date,country,region,cov):
    CovT = []
    Cword = []
    Special = []
    for a in range (0,len(date)):
        #print(country.iloc[a])
        #print(region.iloc[a])
        #print(cov.iloc[a])
        #print(country[a])
        #print(region[a])
        #print(cov[a])
        year = int(date[a][0:4])
        mo = int(date[a][4:6])
        dy = int(date[a][6:8])
        if year==2020:
            DaysOut = dy - 20
        else:
            DaysOut = np.sum(DIM[:(mo-1)])+dy+11
        found = 0
        for b in range (0,len(halo_country)):
            if halo_country[b]==country.iloc[a] and (halo_region[b]==region.iloc[a] or (halo_region[b]=='all' and region.iloc[a]=='')):
                HALOpert = []
                found += 1
                for c in range (0,len(halo_p[b])):
                    HALOpert.append(float(halo_p[b][c]))
        #print(found,country[a])
        if found == 0:
            HALOpert = [0,0,0,0,0,0,0,0]
        UltPert = np.interp(HALOdyall,HALOdy,HALOpert)
        CovT.append((UltPert[int(DaysOut)]*SF*(cov.iloc[a]*0.07))+cov.iloc[a])
        # if a<300:
        #     print(cov.iloc[a],UltPert[int(DaysOut)],((UltPert[int(DaysOut)]*SF*(cov.iloc[a]*0.1))+cov.iloc[a]),"T1")
        ll = intlseed+a
        ul = ll+3
        unum = LOnum[ll:ul]
        LOidx = (((unum[0]*256) + unum[1] ) * ((unum[2]%2)+1))
        if a%8==0:
            Cword.append(C_Blog[LOidx])
        if a%8==1:
            Cword.append(C_Web[LOidx])
        if a%8==2:
            Cword.append(C_TV[LOidx])
        if a%8==3:
            Cword.append(C_Spoken[LOidx])
        if a%8==4:
            Cword.append(C_Fiction[LOidx])
        if a%8==5:
            Cword.append(C_Magazine[LOidx])
        if a%8==6:
            Cword.append(C_News[LOidx])
        if a%8==7:
            Cword.append(C_Academic[LOidx])
        found = 0
        for b in range (0,len(ISPctry)):
            if ISPctry[b]==country.iloc[a] and ((ISPstate[b]==region.iloc[a]) or (ISPstate[b]=='all' and region.iloc[a]=='')):
                Special.append(SPQ[b])
                found += 1
        if found == 0:
            print(country.iloc[a], region.iloc[a])
            Special.append(0)

    #US full/statesum = 1.0042472748866385
    #UK full/statesum = 0.9730115693235666
    #print(len(Special),len(CovT),len(Cword),len(date))
    for a in range (0,len(date)):
        Lmsg = 'fire0'
        if country.iloc[a]=='United Kingdom' and (region.iloc[a]=='all' or region.iloc[a]==''):
            Lmsg='fire1'
            statesum=0
            for b in range (0,len(date)):
               if date[a]==date[b] and country.iloc[b]=='United Kingdom' and region.iloc[b]!='all' and region.iloc[b]!='':
                    statesum += CovT[b]
            CovT[a] = statesum*0.973012
        if country.iloc[a]=='United States' and (region.iloc[a]=='all' or region.iloc[a]==''):
            Lmsg='fire2'
            statesum=0
            for b in range (0,len(date)):
               if date[a]==date[b] and country.iloc[b]=='United States' and region.iloc[b]!='all' and region.iloc[b]!='':
                    statesum += CovT[b]
            CovT[a] = statesum*1.004247
        
        # if a<300:
        #     print(CovT[a],Lmsg,Cword[a])
        
    return CovT,Special,Cword