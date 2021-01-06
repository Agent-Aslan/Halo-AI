import csv
from urllib.request import urlopen as uReq
import numpy as np
import scipy.stats
#import geopandas
import matplotlib.pyplot as plt
import sys
import serial
from serial.tools import list_ports
import time

UseTrueRNG = True
HALO = True
NEDspeed = 250#max = 50000
TurboSpeed = 250# max = 400000

CoThres = 0.5
NodeThres = 0.3141

outfile_path = 'C:/Users/Aslan/HALO'
starttime = time.time()

outfile = open('%s/CovModel_%d.txt'%(outfile_path,starttime),'w')
modfile = open('%s/CovModel_modfile_%d.txt'%(outfile_path,starttime),'w')


if UseTrueRNG==True:

    
    
    ports=dict()  
    ports_avaiable = list(list_ports.comports())
    
    
    rngcomports = []
    turbocom = None
    
    for temp in ports_avaiable:
        if HALO==True:
        	if temp[1].startswith("TrueRNG"):
        		if 'pro' in temp[1]:
        			print ('found pro')
        			turbocom = str(temp[0])
        		else:
        			print('Found:           ' + str(temp))
        			rngcomports.append(str(temp[0]))
        else:
        	if temp[1].startswith("TrueRNG"):
        		print ('found device')
        		turbocom = str(temp[0])
            
    if HALO==True:
        ser = []            
        for a in range(0,len(rngcomports)):
        	ser.append (serial.Serial(port=rngcomports[a],timeout=10))           
    turboser= (serial.Serial(port=turbocom,timeout=10)) 
    
    
               
    #print('Using com port:  ' + str(rng1_com_port))
    #print('Using com port:  ' + str(rng2_com_port))
    #print('==================================================')
    sys.stdout.flush()
    
    if HALO==True:
        for a in range(0,len(rngcomports)):
        	if(ser[a].isOpen() == False):
        		ser[a].open()
        
        	ser[a].setDTR(True)
        	ser[a].flushInput()
    if turboser.isOpen()==False:
        turboser.open()
    turboser.setDTR(True)
    turboser.flushInput()
    
    
    
    
    sys.stdout.flush()

def CohSampMain(params,Zthres,minruns):
    TotalRuns=0
    Zval = 0
    bitct=[]
    pct = []
    allnodes=[]
    for a in range (0,params):
        pct.append([])
    bads = np.zeros(params)
    while np.abs(Zval)<Zthres or TotalRuns<minruns:
        turboser.flushInput()
        supernode = turboser.read(TurboSpeed)
        
        for b in range (0,len(supernode)):
            outfile.write('%d,'%(supernode[b]))
            allnodes.append(supernode[b])
            strnode = str(bin(256+int(supernode[b])))[3:]
            bitct.append(int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7]))
        outfile.write('%d,T\n'%(int(time.time()*1000)))
        
        
        for a in range(0,params):
            if HALO==True:
                ser[a].flushInput()         
                node = ser[a].read(NEDspeed)
            else:
                node = turboser.read(NEDspeed)
            #print (a,len(node),TotalRuns)
            if len(node)==0:
                print('BAD READ ON %s'%rngcomports[a])
                bads[a] += 1
            else:
                for mm in range (0,NEDspeed):
                    outfile.write('%d,'%(node[mm]))
                    strnum = bin(256+node[mm])[3:]
                    pct[a].append((int(strnum[0]) + int(strnum[1]) + int(strnum[2]) + int(strnum[3]) + int(strnum[4]) + int(strnum[5]) + int(strnum[6]) + int(strnum[7])))
            outfile.write('%d,%d\n'%(int(time.time()*1000),a))
        
        
        if TotalRuns < 300:
            NedVal = np.sum(bitct)
            TotalRuns += 1
            #print(bitct)
        else:
            TotalRuns = 300
            NedVal = np.sum(bitct[-(300*TurboSpeed):])
            #print(bitct[-60:])
        
        EX = NedVal-(TotalRuns*TurboSpeed*8*0.5)
        snpq = (TotalRuns*TurboSpeed*8*0.25)**0.5
        #print(TotalRuns,NedVal,EX,snpq)
        Zval = EX/snpq
        #print(Zval)
        
        Z=[]
        N = TotalRuns*NEDspeed*8
        for a in range (0,params):
            if TotalRuns < 300:
                NedVal_x = np.sum(pct[a])
            else:
                NedVal_x = np.sum(pct[a][-(300*NEDspeed):])
            Z.append((NedVal_x - (N*0.5)) / ((N*0.25)**0.5))
        
        time.sleep(0.2)
    #print(Z)
    #print(pct)
    #print(N)
    return Z,allnodes[(-minruns*TurboSpeed):]

#red = geopandas.datasets.get_path('c_10nv20')
#world = geopandas.read_file(red)
#rworld = world.iloc

FIPS = []
LAT = []
LON = []
readFile = open('%s/FIPSout.txt'%outfile_path,'r')
sepfile = readFile.read().split('\n')
for a in range (0,len(sepfile)):
    xandy = sepfile[a].split(',')
    FIPS.append(int(xandy[0]))
    LAT.append(float(xandy[1]))
    LON.append(float(xandy[2]))

"""
for a in range (0,3323):
    FIPS.append(int(rworld[a][3]))
    LAT.append(float(rworld[a][7]))
    LON.append(float(rworld[a][6]))
    
outfile = open('K:/FIPSout.txt','w')
for a in range (0,len(FIPS)):
    outfile.write('%d,%f,%f\n'%(FIPS[a],LAT[a],LON[a]))
outfile.close()
"""

#counties = world.split('\n')
#print(len(rworld))
#print((rworld[3323]))
#geopandas.read_file('K:/counties/c_10nv20')


CoPop=[]
readFile = open('%s/CoPop.txt'%outfile_path,'r')
sepfile = readFile.read().split('\n')
for a in range (0,len(sepfile)):
    CoPop.append(int(sepfile[a]))


COVdat = 'http://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv'
uClient = uReq(COVdat)
page_csv = str(uClient.read())
uClient.close()

sepfile = page_csv.split('\\r\\n')




ult_Q=[]
ult_pop=[]
ult_lat=[]
ult_lon=[]
ult_FIPS=[]
for a in range (2,len(sepfile)):
    
    if CoPop[a-2]>0:
        xandy = sepfile[a].split(',')
        cfips = int(xandy[0])
        ct=0
        for b in range (0,len(FIPS)):
            if FIPS[b]==cfips and ct==0:
                ult_lat.append(LAT[b])
                ult_lon.append(LON[b])
                ult_FIPS.append(FIPS[b])
                ct += 1
        
        if ct==1:
        
            xc = xandy[4:]
            
            if '"' in xc[-1]:
                d = int(xc[-1][:-1])
                
            else:
                d = int(xc[-1])
            m14 = int(xc[-14])
            
            ult_Q.append((d-m14)/CoPop[a-2])
            ult_pop.append(CoPop[a-2])
        
print(len(ult_Q),len(ult_lat))
 
MeanQ = np.nanmean(ult_Q)
stdQ = np.nanstd(ult_Q)   
ult_QZ = (np.array(ult_Q)-MeanQ)/stdQ
ult_pop_log = np.log10(np.array(ult_pop))
        
RankQ = scipy.stats.rankdata(ult_Q)


"""
Sig=[]
for a in range (0,len(ult_Q)):
    sx = np.random.randint(0,2,1000)
    Sig.append((np.sum(sx)-500)/((1000*0.25)**0.5))
RankS = scipy.stats.rankdata(Sig)
"""

def Cdist(lat1,lon1,lat2,lon2):
    yy = (lat2-lat1)*111
    xx = (lon2-lon1)*np.cos(((lat1+lat2)/2)*(np.pi/180))*111
    dist = (((xx**2)+(yy**2))**0.5)
    return dist

ult_pert=[]
OnlyNodesLat=[]
OnlyNodesLon=[]
OnlyNodesPert=[]
NodeCt = 0
for a in range (0,len(ult_pop)):
    sel = CohSampMain(1,CoThres,50)[1]
    nnode = ((sel[-3]%16)*(16**4)) + (sel[-2]*256) + (sel[-1])
    
    #nnode = np.random.randint(0,2**20)
    if nnode <= ult_pop[a]:#if number generated exceeds probabilistic threshold, we make the county a node
        if len(OnlyNodesLat)>0:#
            nDist=[]
            for b in range (0,len(OnlyNodesLat)):
                nDist.append(Cdist(OnlyNodesLat[b],OnlyNodesLon[b],ult_lat[a],ult_lon[a]))
            if np.nanmin(nDist) > 100:#can only be a node if no previous nodes within 50 km
                #sx = np.random.randint(0,2,1000)
                #Pur = (np.sum(sx)-500)/((1000*0.25)**0.5)
                print('working on node %d of about 130'%NodeCt)
                sx = CohSampMain(1,NodeThres,300)
                print('node complete')
                Pur = sx[0][0]
                sxx = sx[1]
                for c in range (0,len(sxx)):
                    modfile.write('%d,'%(sxx[c]))
                modfile.write('%d,%d,%f\n'%(time.time()*1000,ult_FIPS[a],Pur))
                    
                
                ult_pert.append(Pur)
                NodeCt += 1
                OnlyNodesLat.append(ult_lat[a])
                OnlyNodesLon.append(ult_lon[a])
                OnlyNodesPert.append(Pur)
            else:
                ult_pert.append(-9999)
        else:
            #sx = np.random.randint(0,2,1000)
            #Pur = (np.sum(sx)-500)/((1000*0.25)**0.5)
            print('working on node %d of about 130'%NodeCt)
            sx = CohSampMain(1,NodeThres,300)
            print('node complete')
            Pur = sx[0][0]
            sxx = sx[1]
            for c in range (0,len(sxx)):
                modfile.write('%d,'%(sxx[c]))
            modfile.write('%d,%d,%f\n'%(time.time()*1000,ult_FIPS[a],Pur))
            
            
            
            ult_pert.append(Pur)
            NodeCt += 1
            OnlyNodesLat.append(ult_lat[a])
            OnlyNodesLon.append(ult_lon[a])
            OnlyNodesPert.append(Pur)
    else:
        ult_pert.append(-9999)
        
    if a%10==0:
        print('county %d of %d complete'%(a,len(ult_pop)))
        
print(NodeCt)




ip_pert=[]
Rsave=[]
for a in range (0,len(ult_pert)):
    if ult_pert[a]==-9999:
        ff_weights=[]; ff_sig=[]
        for b in range (0,len(ult_pert)):
            if ult_pert[b]>-9999:
                r2 = (Cdist(ult_lat[a],ult_lon[a],ult_lat[b],ult_lon[b])**2)
                Rsave.append(r2)
                ff_weights.append(1/r2)
                ff_sig.append(ult_pert[b])
        ip_pert.append(np.average(ff_sig,weights=ff_weights))
    else:
        ip_pert.append(ult_pert[a])
                

RankS = scipy.stats.rankdata(ip_pert)


co_R=[]
co_S=[]
for a in range (0,len(RankS)):
    if ult_pert[a]>-9999:
        #co_R.append(RankQ[a])
        #co_S.append(RankS[a])
        co_R.append(ult_QZ[a])
        co_S.append(ip_pert[a])
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
#ax1.scatter(ult_lon,ult_lat,c=RankS,cmap='seismic')
ax1.scatter(ult_lon,ult_lat,c=ip_pert,cmap='seismic',vmin=-2,vmax=2)
ax1.scatter(OnlyNodesLon,OnlyNodesLat,c=OnlyNodesPert,marker = '^',cmap = 'seismic',vmin=-2,vmax=2)#what we're showing is RANK so the flip isn't quite clear around some points. Needs to be rank?   Further its rank Q and not rank QZ
#ax2.scatter(ult_lon,ult_lat,c=RankQ,cmap='seismic')
ax2.scatter(ult_lon,ult_lat,c=ult_QZ,cmap='seismic',vmin=-2,vmax=2)
ax3.scatter(co_R,co_S)
plt.show()

MidS = np.median(co_S)
MidQ = np.median(co_R)
hit = 0
for a in range (0,len(co_S)):
    if (co_S[a]>MidS and co_R[a]>MidQ) or (co_S[a]<MidS and co_R[a]<MidQ):
        hit += 1

nZ = len(co_S)
Z = (hit-(nZ*0.5))/((nZ*0.25)**0.5)

print('hits: %d, total: %d, Z: %f'%(hit,nZ,Z))
#do NOT want to weight this score by population, nodes in rural areas are important as they speak to same population broader area. Also this is a case against variable A0 based on node population. Just ML hits and misses. And where note to do this for WRF as well? (can do RETROACTIVELY and see if works. As can with LO with meaning applied. etc. document this.)
#correlate only nodes. A0 based on log(pop).
#visual check binomal hit/miss with this. A0 + previous data learn and look for that as new point.(after 2w)...so ML model applied after first .  A0~pop




outfile.close()
modfile.close()