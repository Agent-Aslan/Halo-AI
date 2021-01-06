#!/usr/bin/env python
# coding=utf-8
import os
import sys
import serial
import time
from time import sleep
#import sched
#from serial import Serial
from serial.tools import list_ports
import numpy as np
import scipy.stats
import pyttsx3#must be version 2.6

import matplotlib.pyplot as plt
import matplotlib.animation as animation

NEDspeed = 1000#max = 50000
TurboNEDspeed = 10000# max = 400000
UseTurbo = True
NedThreshold = 1.239
MinTime = 2

ports=dict()  
ports_avaiable = list(list_ports.comports())


#####LO######
Blog=[]
Web=[]
TV=[]
Spoken=[]
Fiction=[]
Magazine=[]
News=[]
Academic=[]


Word=[]
Readfile=open('C:/Users/Aslan/Documents/HALO Development/LanguageDB/words_219k_s3514.txt','r')
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

"""
s1=sorted(Blog, reverse=True)
s2=sorted(Web, reverse=True)
s3=sorted(TV, reverse=True)
s4=sorted(Spoken, reverse=True)
s5=sorted(Fiction, reverse=True)
s6=sorted(Magazine, reverse=True)
s7=sorted(News, reverse=True)
s8=sorted(Academic, reverse=True)
i = 65536*2
print(s1[i],s2[i],s3[i],s4[i],s4[i],s5[i],s6[i],s7[i],s8[i])
"""

#sBlog=sorted(Blog,reverse=True)
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

# C_General = Word[0:OutputLength]
C_Blog=sBlogWord[0:131072]
C_Web=sWebWord[0:131072]
C_TV=sTVWord[0:131072]
C_Spoken=sSpokenWord[0:131072]
C_Fiction=sFictionWord[0:131072]
C_Magazine=sMagazineWord[0:131072]
C_News=sNewsWord[0:131072]
C_Academic=sAcademicWord[0:131072]

#DoNotUse = ['o','mg','v','ta','g','p','obama','qwq','trump','donald' 'et','m','clinton','nt','h','republicans','democrats','senate','cnn','abc','oct','roker','c','r','j','n','o','p','b','a','d','e','f','g','pp','x','y','z','hillary']

"""
AllWords=[]
for A in range(0,len(Word)):
    if (sBlogWord[A] not in C_General) and (len(C_Blog)<OutputLength) and (sBlogWord[A] not in DoNotUse):      
        C_Blog.append(sBlogWord[A])
        AllWords.append(sBlogWord[A])
for A in range(0,len(Word)):
    if (sWebWord[A] not in C_General) and (sWebWord[A] not in AllWords) and (len(C_Web)<OutputLength) and (sWebWord[A] not in DoNotUse):      
        C_Web.append(sWebWord[A])
        AllWords.append(sWebWord[A])
for A in range(0,len(Word)):
    if (sTVWord[A] not in C_General) and (sTVWord[A] not in AllWords) and (len(C_TV)<OutputLength) and (sTVWord[A] not in DoNotUse):      
        C_TV.append(sTVWord[A])
        AllWords.append(sTVWord[A])
for A in range(0,len(Word)):
    if (sSpokenWord[A] not in C_General) and (sSpokenWord[A] not in AllWords) and (len(C_Spoken)<OutputLength) and (sSpokenWord[A] not in DoNotUse):      
        C_Spoken.append(sSpokenWord[A])
        AllWords.append(sSpokenWord[A])
for A in range(0,len(Word)):
    if (sFictionWord[A] not in C_General) and (sFictionWord[A] not in AllWords) and (len(C_Fiction)<OutputLength) and (sFictionWord[A] not in DoNotUse):      
        C_Fiction.append(sFictionWord[A])
        AllWords.append(sFictionWord[A])
for A in range(0,len(Word)):
    if (sMagazineWord[A] not in C_General) and (sMagazineWord[A] not in AllWords) and (len(C_Magazine)<OutputLength) and (sMagazineWord[A] not in DoNotUse):      
        C_Magazine.append(sMagazineWord[A])
        AllWords.append(sMagazineWord[A])
for A in range(0,len(Word)):
    if (sNewsWord[A] not in C_General) and (sNewsWord[A] not in AllWords) and (len(C_News)<OutputLength) and (sNewsWord[A] not in DoNotUse):      
        C_News.append(sNewsWord[A])
        AllWords.append(sNewsWord[A])
      
for A in range(0,len(Word)):
    if (sAcademicWord[A] not in C_General) and (sAcademicWord[A] not in AllWords) and (len(C_Academic)<OutputLength) and (sAcademicWord[A] not in DoNotUse):      
        C_Academic.append(sAcademicWord[A])
        AllWords.append(sAcademicWord[A])
"""
        
print('language oracle built succesfully')     

   
#######LO#######




rngcomports = []
turbocom = None

for temp in ports_avaiable:
    if UseTurbo==True:
    	if temp[1].startswith("TrueRNG"):
    		if 'pro' in temp[1]:
    			print ('found pro')
    			turbocom = str(temp[0])
    		else:
    			print('Found:           ' + str(temp))
    			rngcomports.append(str(temp[0]))
    else:
    	if temp[1].startswith("TrueRNG"):
    		if 'pro' not in temp[1]:
    			print('Found:           ' + str(temp))
    			rngcomports.append(str(temp[0]))
        
           
ser = []            
for a in range(0,len(rngcomports)):
	ser.append (serial.Serial(port=rngcomports[a],timeout=10))           
turboser= (serial.Serial(port=turbocom,timeout=10)) 


           
#print('Using com port:  ' + str(rng1_com_port))
#print('Using com port:  ' + str(rng2_com_port))
#print('==================================================')
sys.stdout.flush()

for a in range(0,len(rngcomports)):
	if(ser[a].isOpen() == False):
		ser[a].open()

	ser[a].setDTR(True)
	ser[a].flushInput()
if UseTurbo==True:
    if turboser.isOpen()==False:
        turboser.open()
    turboser.setDTR(True)
    turboser.flushInput()




sys.stdout.flush()


starttime = int(time.time()*1000)

outfile = open('C:/Users/Aslan/Documents/HALO Development/NED_Data/LO_Test/RNG_%d.txt'%(starttime),'w')
outfileL = open('C:/Users/Aslan/Documents/HALO Development/NED_Data/LO_Test/Words_%d.txt'%(starttime),'w')

#outfileJava = open('C:/Users/Aslan/Documents/HALO Development/NED_Java/RNG_%d.txt'%(starttime),'w')

outfile.write('NED Threshold = %f, Min Time = %f'%(NedThreshold,MinTime))


NedVal = np.zeros(len(rngcomports)+1)

print("Breathing...")
TotalRuns=0


CumZ=[]
CumES=[]
ArcSaveTotal = []
RunTime = []
Zcum=[]
PSQ=[]  
ProSQ=[]  
NedSQ=[]
NormalBits=[]

XOR_T=[]
XorSQ=[]
        
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ProBits=[]
#words = 'Welcome aboard the Hypercube Algorithmic Language Oracle, Your pilot is Captain Madrid and your first officer is Dr. Caputi. Hold on as we chill the fuck out and take a digial trip through the inner space halo travel station, guided by the priciple laws of Anthromurmuration and Spatial Relativity. Be sure to look into the AEtherspheric Modulator for a consciousness expanding experience. '
words = 'Welcome to Scream for Peace, where spontaneous creativity rapidly enhances aetherspheric modulation for people evolve as consciousness expands. We are broadcasting from the Hypercube Algorithmic Language Oracle, Your pilot is Captain Madrid and your first officer is Dr. Caputi. Hold on as we chill the frack out and take a digial trip through the inner space travel station, guided by the priciple laws of Anthromurmuration and Spatial Relativity. Be sure to gaze into the AEtherspheric Modulator for a consciousness expanding experience. '
Gwords = [words]
word_time = [time.time()]

def Speak(txt):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    rate = engine.getProperty('rate')   # getting details of current speaking rate
    engine.setProperty('rate', 131)     # setting up new voice rate
    
    engine.say('%s'%(txt))
    engine.runAndWait()
    engine.stop()


Speak(words)

def MatchCount(stream1,stream2):
    """
    #bytes are input
    lstr = len(streams[0])
    for a in range (0,len(lstr)):
        down = []
        for b in range (0,len(streams)):
            down.append(streams[b][a])
    """
    matchct = 0
    #print(len(stream1),len(stream2))
    
    for a in range (0,len(stream1)):
        strnode1 = str(bin(256+int(stream1[a])))[3:]
        strnode2 = str(bin(256+int(stream2[a])))[3:]
        for b in range (0,8):
            if int(strnode1[b])==int(strnode2[b]):
                matchct += 1
    
    #for a in range (0,len(stream1)):
    #    if (stream1[a]==stream2[a]):
    #        matchct += 1
    #        #print(stream1[a],stream2[a])
    return matchct
    
def Symmetric(stream):
    cmb1 = MatchCount(stream[0],stream[7])
    cmb2 = MatchCount(stream[1],stream[6])
    cmb3 = MatchCount(stream[2],stream[5])
    cmb4 = MatchCount(stream[3],stream[4])
    #cmbT = MatchCount(stream[8][0:1000],stream[8][1000:2000])
    #print(cmb1,cmb2,cmb3,cmb4)
    return cmb1+cmb2+cmb3+cmb4#+cmbT



def animate(i):
    insym = []
    for a in range(0,len(rngcomports)):
        try:
            ser[a].flushInput()      
            node = ser[a].read(NEDspeed)
        except:
            print('%s failure'%(rngcomports[a]))
            node = node.fill(-999)
        insym.append(node)
        bitct=0
        for b in range (0,len(node)):
            outfile.write('%d,'%(node[b]))
            strnode = str(bin(256+int(node[b])))[3:]
            bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])
        outfile.write('%d,%d\n'%(int(time.time()*1000),a))
        NormalBits.append(bitct)
        NedVal[a] += bitct
        #print(bitct)
    if UseTurbo==True:
        turboser.flushInput()
        supernode = turboser.read(TurboNEDspeed)
        bitct=0
        for b in range (0,len(supernode)):
            outfile.write('%d,'%(supernode[b]))
            strnode = str(bin(256+int(supernode[b])))[3:]
            bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])
        ProBits.append(bitct)
    else:
        ProBits.append(0)
    current_time = time.time()
    outfile.write('%d,T\n'%(int(current_time*1000)))
    outfile.flush()
    os.fsync(outfile.fileno())
    #outfileJava.write('[%d %d %d %d %d %d %d %d]\n'%(int(strnode[0]),int(strnode[1]),int(strnode[2]),int(strnode[3]),int(strnode[4]),int(strnode[5]),int(strnode[6]),int(strnode[7])))
    #outfileJava.flush()
    #os.fsync(outfileJava.fileno())
    NedVal[-1] += bitct
    
    #XOR part:
    XOR_L = Symmetric(insym)
    #print(XOR_L)
    XOR_T.append(XOR_L)
    
    
    #TotalRuns += 1
    RunTime.append((current_time%86400)/3600)
    
    Pcum=[]
    for a in range (0,len(NedVal)-1):
        EX = NedVal[a]-(len(RunTime)*NEDspeed*8*0.5)
        #print((TotalRuns*NEDspeed*8*0.5),NedVal[a])
        snpq = (len(RunTime)*NEDspeed*8*0.25)**0.5
        Zval = EX/snpq
        #print(Zval)
        Zcum.append(Zval**2)
        #pval = scipy.stats.norm.sf(np.abs(Zval)*2)
        #Pcum.append(1/pval)
    
    
    
    #MaxP = np.amax(Pcum)
    
    #EX = NedVal[-1]-(TotalRuns*TurboNEDspeed*8*0.5)
    #snpq = (TotalRuns*TurboNEDspeed*8*0.25)**0.5
    #Zval = EX/snpq
    #pval = scipy.stats.norm.sf(np.abs(Zval)*2)
    #print(Pcum,1/pval)
    
    
    #print('%d | %.2f | %.2f | %.2f'%(TotalRuns,1/CmbP,MaxP,1/pval))
    
    
    #CumZ.append(Zval)
    
    #if TotalRuns%600==0:
    #    outfileJava.close()
    #    outfileJava = open('C:/Users/Aslan/Documents/HALO Development/NED_Java/RNG_%d.txt'%(int(time.time()*1000)),'w')
    CCalInt = (len(NedVal)-1)
    CalInt = CCalInt*60
    
    if len(RunTime)>=65:
        PSQ.append(1.0/(scipy.stats.chi2.sf(np.sum(Zcum[-CalInt:]),CalInt)))
        
        NedSQ.append(((np.sum(NormalBits[-60*CCalInt:])) - (0.5*NEDspeed*8*CCalInt*60))/((60*CCalInt*NEDspeed*8*0.25)**0.5))
        ProSQ.append(((np.sum(ProBits[-60:])) - (0.5*TurboNEDspeed*8*60))/((60*TurboNEDspeed*8*0.25)**0.5))
        XorSQ.append(((np.sum(XOR_T[-60:])) - (0.5*NEDspeed*8*4*60))/((60*NEDspeed*8*4*0.25)**0.5))
        
        #print(np.sum(ProBits[-60:]))
    else:
        PSQ.append(0)
        ProSQ.append(0)
        NedSQ.append(0)
        XorSQ.append(0)
        
    #if (XorSQ[-1]**2)+(ProSQ[-1]**2) >= 5
    #print(np.abs(XorSQ[-1]),np.abs(ProSQ[-1]),current_time-word_time[-1])
    if np.abs(XorSQ[-1])>=NedThreshold and np.abs(ProSQ[-1])>=NedThreshold and (current_time-word_time[-1])>=MinTime:
        if ((len(Gwords)%10)-1)==0:
            idx1.append(((supernode[-24]*256) + supernode[-23] ) * ((supernode[-22]%2)+1))
            idx2.append(((supernode[-21]*256) + supernode[-20] ) * ((supernode[-19]%2)+1))
            idx3.append(((supernode[-18]*256) + supernode[-17] ) * ((supernode[-16]%2)+1))
            idx4.append(((supernode[-15]*256) + supernode[-14] ) * ((supernode[-13]%2)+1))
            idx5.append(((supernode[-12]*256) + supernode[-11] ) * ((supernode[-10]%2)+1))
            idx6.append(((supernode[-9]*256) + supernode[-8] ) * ((supernode[-7]%2)+1))
            idx7.append(((supernode[-6]*256) + supernode[-5] ) * ((supernode[-4]%2)+1))
            idx8.append(((supernode[-3]*256) + supernode[-2] ) * ((supernode[-1]%2)+1))
            
            
            
            HaloVoice = '%s %s %s %s %s %s %s %s'%(C_Blog[idx1[-1]],C_Web[idx2[-1]],C_TV[idx3[-1]],C_Spoken[idx4[-1]],C_Fiction[idx5[-1]],C_Magazine[idx6[-1]],C_News[idx7[-1]],C_Academic[idx8[-1]])
            
            
        else:
            if len(Gwords)%8 == 0:
                idx1.append(((supernode[-24]*256) + supernode[-23] ) * ((supernode[-22]%2)+1))
                HaloVoice = '%s'%C_Blog[idx1[-1]]
            if len(Gwords)%8 == 1:
                idx2.append(((supernode[-21]*256) + supernode[-20] ) * ((supernode[-19]%2)+1))
                HaloVoice = '%s'%C_Web[idx2[-1]]
            if len(Gwords)%8 == 2:
                idx3.append(((supernode[-18]*256) + supernode[-17] ) * ((supernode[-16]%2)+1))
                HaloVoice = '%s'%C_TV[idx3[-1]]
            if len(Gwords)%8 == 3:
                idx4.append(((supernode[-15]*256) + supernode[-14] ) * ((supernode[-13]%2)+1))
                HaloVoice = '%s'%C_Spoken[idx4[-1]]
            if len(Gwords)%8 == 4:
                idx5.append(((supernode[-12]*256) + supernode[-11] ) * ((supernode[-10]%2)+1))
                HaloVoice = '%s'%C_Fiction[idx5[-1]]
            if len(Gwords)%8 == 5:
                idx6.append(((supernode[-9]*256) + supernode[-8] ) * ((supernode[-7]%2)+1))
                HaloVoice = '%s'%C_Magazine[idx6[-1]]
            if len(Gwords)%8 == 6:
                idx7.append(((supernode[-6]*256) + supernode[-5] ) * ((supernode[-4]%2)+1))
                HaloVoice = '%s'%C_News[idx7[-1]]
            if len(Gwords)%8 == 7:
                idx8.append(((supernode[-3]*256) + supernode[-2] ) * ((supernode[-1]%2)+1))
                HaloVoice = '%s'%C_Academic[idx8[-1]]
        
        
        words = '%s || %s || %s || %s || %s || %s || %s || %s'%(C_Blog[idx1[-1]],C_Web[idx2[-1]],C_TV[idx3[-1]],C_Spoken[idx4[-1]],C_Fiction[idx5[-1]],C_Magazine[idx6[-1]],C_News[idx7[-1]],C_Academic[idx8[-1]])
        Speak(HaloVoice)
        Gwords.append(words)
        outfileL.write('%d %s\n'%(current_time,words))
        outfileL.flush()
        os.fsync(outfileL.fileno())
        word_time.append(current_time)
        
        #print('inside act')
    #print(NedVal)                                                          
    ax1.clear()
    ax2.clear()
    
    #ax1.plot(RunTime,NedSQ)
    ax1.plot(RunTime,XorSQ)
    ax1.set_ylabel('Normal Z')
    ax2.plot(RunTime,ProSQ)
    ax2.set_ylabel('Turbo Z')
    fig.suptitle('%s'%Gwords[-1])

idx1=[]
idx2=[]
idx3=[]
idx4=[]
idx5=[]
idx6=[]
idx7=[]
idx8=[]

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()



