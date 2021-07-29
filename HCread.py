# This import registers the 3D projection, but is otherwise unused.
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.animation as animation
#import os
import sys
#import serial
#from serial.tools import list_ports
#import time
import math
import scipy.stats
#import matplotlib.gridspec as gridspec
import warnings
#from matplotlib.widgets import Button
#import tkinter as tk
#from astral import LocationInfo
#import datetime
#from astral.sun import sunrise,sunset
warnings.simplefilter('ignore')


#auto detection, common words aem clock
#with lightbox etc fine...but ALSO HERE have color coherence. Data Z trend modules too and cross cmp times.    color coherence p cum trend lk over time when have code lv back or remake here
#Milti file chisq
#p cum... and full data strawberry both there here.
#readable dts


#rngs = 4
#NEDspeed = 250
#TurboUse = False
#ColorZ = 1.65
#RotZ = 1.85
outpath = 'E:/PeaceIncExperiments/Formal_Experiments/Strawberry Moon 2021'

anlzfile = sys.argv[1]


readFile = open('%s/%s'%(outpath,anlzfile),'r')
sepfile = readFile.read().split('\n')
head = sepfile[0].split(' ')
Type = head[1]
ColorZ = float(head[3])
RotZ = float(head[5])
firstline = sepfile[1].split(',')
if firstline[-1]=='T':
    TurboUse = True
else:
    TurboUse = False
NEDspeed = len(firstline)-2
rngs = 1
rngct = 0
while rngct==0:
    line = sepfile[rngs+1]
    if 'QBYTE' in line:
        rngct += 1
    else:
        rngs += 1
        
if Type == 'hypercube':
    infile = 'SimulationsHC.txt'
if Type == 'sphere' or Type == 'nye':
    infile = 'Simulations_Sphere12.txt'
if Type=='AEM' or Type=='AEMclock':
    infile = 'Simulations_AEM10.txt'
        
print(rngs,NEDspeed,TurboUse,ColorZ,RotZ)#next is auto input readM



ActionNumC = math.ceil((ColorZ*((8*NEDspeed*0.25)**0.5))+(4*NEDspeed))
Pmod_Color = (scipy.stats.binom((NEDspeed*8),0.5).sf(ActionNumC-1))*2
ActionNumR = math.ceil((RotZ*((8*NEDspeed*0.25)**0.5))+(4*NEDspeed))
Pmod_Rot = (scipy.stats.binom((NEDspeed*8),0.5).sf(ActionNumR-1))*2



SimMI = []    
readFileM = open('%s/%s'%(outpath,infile),'r')
sepfileM = readFileM.read().split('\n')
for a in range (0,len(sepfileM)):
    SimMI.append(float(sepfileM[a]))
Msorted = sorted(SimMI)

def M2P(MIv):
    idx = 0
    for a in range (0,len(Msorted)):
        if Msorted[a]>MIv:
            idx += 1
    if idx==0:
        idx += 1
    p_i = idx/1000000
    #print(idx,MIv,np.amin(Msorted),np.amax(Msorted))
    return p_i




ult_t=[]
ult_sums=[]
ax1y=[]
for a in range (0,rngs+1):
    ult_sums.append([])
    ax1y.append([])

ax1s=[]
ax1sN=[]

MIt=[]
MIp=[]
CumP=[]

colorsave=[]
rotsave=[]

colorct=0
rotct = 0

Mplt=[]
Mstd=[]
Rplt=[]
Rstd=[]

KMlog = 0

for a in range (0,len(sepfile)):
    if 'QBYTE' in sepfile[a]:#goal: get bitsums. EE can be read. hopefully retrieve code for good read_lights code. so yeah anlz dep on LV..
              
        bitct = 0
        xandy = sepfile[a].split(',')
        for b in range (0,len(xandy)-2):            
            strnode = str(bin(256+int(xandy[b])))[3:]
            bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])
        
        ult_sums[0].append(bitct)
        
        ult_t.append((int(xandy[-2])-1622530800000)/86400000)
        
        for b in range (1,rngs+1):
            bitct = 0
            xandy = sepfile[a-b].split(',')
            for c in range (0,len(xandy)-2):            
                strnode = str(bin(256+int(xandy[c])))[3:]
                bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])
            ult_sums[b].append(bitct)
        colorsave.append(colorct)
        rotsave.append(rotct)
    if 'color' in sepfile[a]:
        MIt.append(ult_t[-1])
        ival = float(sepfile[a].split(',')[1])
        pval = M2P(ival)
        MIp.append(1/(pval))
        
        KMlog += np.log(pval)
        CumP.append(scipy.stats.chi2.sf((KMlog*-2),(2*len(MIt))))
        
        colorct += 1
        
    if 'rotation' in sepfile[a]:
        rotct += 1
        
    if a%10000==0:
        print("processing line %d of %d"%(a,len(sepfile)))
    
        

for a in range (0,len(ult_t)):
    Nt = (a-(10*colorsave[a]))-11
    Mplt.append(colorsave[a] - (Pmod_Color*Nt))
    Mstd.append(((Nt*Pmod_Color*(1-Pmod_Color))**0.5)*1.65)
    
    NtR = (a-(10*rotsave[a]))-11
    Rplt.append(rotsave[a] - (Pmod_Rot*NtR))
    Rstd.append(((NtR*Pmod_Rot*(1-Pmod_Rot))**0.5)*1.65)

for a in range (0,len(ult_t)):
    for b in range (0,len(ax1y)):
        ax1y[b].append(np.sum(ult_sums[b][0:a])-((a)*NEDspeed*8*0.5))
    ax1s.append((((a)*NEDspeed*8*0.25)**0.5)*1.96)
    ax1sN.append((((a)*NEDspeed*8*0.25)**0.5)*-1.96)
    

plt.style.use('dark_background')
#plt.grid([False])
fig = plt.figure(constrained_layout=True)
#gs = fig.add_gridspec(3,2)
#ax1 = fig.add_subplot(gs[:,0], projection='3d')
ax2 = fig.add_subplot(311)
ax3 = fig.add_subplot(312)
ax4 = fig.add_subplot(313)
    

#Q,T,1234...,
ax2.plot(ult_t,ax1y[0],color='magenta',linewidth='1')
if TurboUse==True:
    ax2.plot(ult_t,ax1y[1],color='red',linewidth='1')
    for a in range (2,len(ax1y)):
        ax2.plot(ult_t,ax1y[a],color='lightgray',linewidth='1')
else:
    for a in range (1,len(ax1y)):
        ax2.plot(ult_t,ax1y[a],color='lightgray',linewidth='1')

ax2.plot(ult_t,ax1s,color='aqua',linestyle='--')
ax2.plot(ult_t,ax1sN,color='aqua',linestyle='--')
    
ax3.plot(ult_t,Mplt,color='aqua')
ax3.plot(ult_t,Mstd,color='aqua',linestyle='--')
ax3.plot(ult_t,Rplt,color='red')
ax3.plot(ult_t,Rstd,color='red',linestyle='--')
ax4.plot(MIt,MIp,color='aqua')
ax4.set_ylabel('Color Coherence 1/p')
ax4.set_yscale('log')
ax4t = ax4.twinx()
ax4t.plot(MIt,CumP,color='red')



    
ax2.set_ylabel('Color Change')
ax3.set_ylabel('Rotation / Color Change')





plt.show()

