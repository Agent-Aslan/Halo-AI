# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import sys
import serial
from serial.tools import list_ports
import time
import math
import scipy.stats
import matplotlib.gridspec as gridspec
import warnings
from matplotlib.widgets import Button
import tkinter as tk
from astral import LocationInfo
import datetime
from astral.sun import sunrise,sunset
warnings.simplefilter('ignore')

#KNOWN ISSUE (12/26/2020 - not fixed): using an odd number of divisiors in sphere shape results in nodes apparently switching on sides of sphere.
#stats = rotateZmod, colorZmod, MI/grid, MI total. can we shrink plot area so 4 params in top to btm and whole left side is HC?/pyramid/whatever?
#not doing max per thousand here.

NEDspeed = 250
ColorZ = 1.65
RotZ = 1.85
outpath = 'C:/Users/Danny/Hypercube6'
DotSize = 4444
UseTrueRNG = True
HALO = True
wordsize = 36
TurboUse = False
FileMax = 100

Type = sys.argv[1]#hypercube,sphere,nye,pyramid,AEM,AEMclock - if nye must input target time as third argument in command line
Rmks = sys.argv[2]#remarks

###########

TurboSpeed = NEDspeed


starttime = int(time.time()*1000)
outfile = open('%s/HYPERCUBE_%s_%d_%s.txt'%(outpath,Type,starttime,Rmks),'w')
cmtfile = open('%s/HYPERCUBE_%s_%d_%s_C.txt'%(outpath,Type,starttime,Rmks),'w')

outfile.write('Type: %s ColorZ: %f RotZ: %f\n'%(Type,ColorZ,RotZ))





DayStarted = starttime - (starttime%86400000)
StartXT = (starttime-DayStarted)/3600000


EX = NEDspeed*4
ColorThres = ColorZ * ((NEDspeed*8*0.25)**0.5)
RotThres = RotZ * ((NEDspeed*8*0.25)**0.5)


ActionNumC = math.ceil((ColorZ*((8*NEDspeed*0.25)**0.5))+(4*NEDspeed))
Pmod_Color = (scipy.stats.binom((NEDspeed*8),0.5).sf(ActionNumC-1))*2
ActionNumR = math.ceil((RotZ*((8*NEDspeed*0.25)**0.5))+(4*NEDspeed))
Pmod_Rot = (scipy.stats.binom((NEDspeed*8),0.5).sf(ActionNumR-1))*2

plt.style.use('dark_background')
#plt.grid([False])
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[:,0], projection='3d')
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,1])
ax4t = ax4.twinx()

city = LocationInfo(latitude=37.7,longitude=-122.2)

riprise=[]
ripset=[]

zoomsto = [1]

for a in range (0,10):

    srise = sunrise(city.observer, date=datetime.datetime.now()+datetime.timedelta(days=a))
    sset = sunset(city.observer, date=datetime.datetime.now()+datetime.timedelta(days=a))
    riprise.append((srise.hour+(srise.minute/60))+(24*a))
    ripset.append((sset.hour+(sset.minute/60))+(24*a))
    

def printInput():
    inp = inputtxt.get(1.0, "end-1c")
    cmtfile.write('%d,%s\n'%(int(time.time()*1000),inp))
    cmtfile.flush()
    os.fsync(cmtfile.fileno())
    frame.wm_withdraw()
    inputtxt.delete(1.0,"end-1c")
    
def kill(self):
    ani.event_source.stop()
    outfile.close()
    cmtfile.close()
    sys.exit()

def comment(self):
    frame.wm_deiconify()
    frame.mainloop()
    
def zoom(self):
    tmp = zoomsto[0]
    if tmp==0:
        zoomsto[0] = 1
    else:
        zoomsto[0] = 0
    
frame = tk.Tk()
frame.title("eh?")
frame.geometry('400x50')
inputtxt = tk.Text(frame,height = 1,width = 35)
inputtxt.pack()
printButton = tk.Button(frame,text = "Submit", command = printInput)
printButton.pack()

axcmt = plt.axes([0.05, 0.05, 0.12, 0.05])
bcmt = Button(axcmt, 'comment', color = '0.5', hovercolor='0.8')
bcmt.on_clicked(comment)

axkil = plt.axes([0.20, 0.05, 0.12, 0.05])
bkil = Button(axkil, 'stop', color = '0.5', hovercolor='0.8')
bkil.on_clicked(kill)

axzoom = plt.axes([0.35, 0.05, 0.12, 0.05])
bzoom = Button(axzoom, 'zoom', color = '0.5', hovercolor='0.8')
bzoom.on_clicked(zoom)


AllLO=[]
Readfile=open('%s/words_219k_s3514.txt'%outpath,encoding='latin-1')
Lines=Readfile.read().split('\n')
for line in range(9,len(Lines)-1):
    items=Lines[line].split('\t')
    AllLO.append(items[1])


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
    if TurboUse==True:
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
    if TurboUse==True:
        if turboser.isOpen()==False:
            turboser.open()
        turboser.setDTR(True)
        turboser.flushInput()
        
        sys.stdout.flush()



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
    
    




def Bulk():
    
    pct = []
    allsums=[]
    for a in range (0,5):
        pct.append([])
        
    if TurboUse==True:
    
        turboser.flushInput()
        supernode = turboser.read(TurboSpeed)#CHG
            
        tempsum = 0
        for b in range (0,len(supernode)):
            outfile.write('%d,'%(supernode[b]))
            pct[4].append(supernode[b])
            
            
            #allnodes.append(supernode[b])
            strnode = str(bin(256+int(supernode[b])))[3:]
            tempsum += (int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7]))
        outfile.write('%d,T\n'%(int(time.time()*1000)))
        
        allsums.append(tempsum)
        
    for a in range(0,4):
        if HALO==True or TurboUse==False:
            try:
                ser[a%len(ser)].flushInput()
                node = ser[a%len(ser)].read(NEDspeed)
            except:
                node = []
        else:
            node = turboser.read(NEDspeed)
        #print (a,len(node),TotalRuns)
        while len(node)==0:
            print('BAD READ ON %s ... removing'%rngcomports[a%len(ser)])
            ser.remove(ser[a%len(ser)])
            #bads[a] += 1
            try:
                ser[a%len(ser)].flushInput()
                node = ser[a%len(ser)].read(NEDspeed)
            except:
                node = []
       
        tempsum = 0
        for mm in range (0,NEDspeed):
            outfile.write('%d,'%(node[mm]))
            strnum = bin(256+node[mm])[3:]
            pct[a].append(node[mm])
            
            strnode = str(strnum)
            tempsum += (int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7]))
        allsums.append(tempsum)
        outfile.write('%d,%s\n'%(int(time.time()*1000),rngcomports[a%len(ser)]))
        
    x = []#CHG should be NEDspeed long
    
    Pur0 = pct[0]
    Pur1 = pct[1]
    Pur2 = pct[2]
    Pur3 = pct[3]
    #Pur4 = pct[4]
    #Pur5 = pct[5]
    #Pur6 = pct[6]
    #Pur7 = pct[7]
    if TurboUse==True:
        PurT = pct[4]
    
    
    for b in range (0,len(Pur0)):
        #xA = Pur0[b]^Pur7[b]
        #xB = Pur1[b]^Pur6[b]
        #xC = Pur2[b]^Pur5[b]
        #xD = Pur3[b]^Pur4[b]
        
        
        
        xE = Pur0[b]^Pur3[b]#xA^xD
        xF = Pur1[b]^Pur2[b]#xB^xC
        
        xG = xE^xF
        
        if TurboUse==True:
        
            x.append(xG^PurT[b])
        else:
            x.append(xG)
        
        
        


    

    
    #OG:
    #ser.flushInput()
    #x = ser.read(NEDspeed)
    
    bitct = 0
    for a in range (0,len(x)):
        outfile.write('%d,'%x[a])
        strnode = str(bin(256+int(x[a])))[3:]
        bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])        
        
    outfile.write('%d,QBYTE\n'%(int(time.time()*1000)))
    
    
    
    outfile.flush()
    os.fsync(outfile.fileno())
    
    
    str0 = str(bin(256+int(x[-2])))[3:] + str(bin(256+int(x[-1])))[3:]
    ones = int(str0[0])+int(str0[1])+int(str0[2])+int(str0[3])+int(str0[4])+int(str0[5])+int(str0[6])+int(str0[7])+int(str0[8])+int(str0[9])+int(str0[10])+int(str0[11])+int(str0[12])+int(str0[13])+int(str0[14])+int(str0[15])

    cat = np.random.randint(0,3)
    LO_idx = ((cat%3)*65536)+(x[-2]*256)+x[-1]
    wrd = AllLO[LO_idx]

    Z = np.abs(ones-8)

    if ones==8:
        symbol = '.0'
    if ones<8:
        symbol = '.-%d'%Z
    if ones>8:
        symbol = '.+%d'%Z
        
    #print(len(x))
    
    return x,bitct,symbol,wrd,allsums

maxon = 65535
def GetColors(colors):
    #colors,r1 = Bulk()
    offset = 0
    Config = []
    U=[]
    V=[]
    for lights in range (0,len(sNode)):
        slider = (colors[-1+offset]*256)+colors[-2+offset]
        uidx = -3+offset
        sector = -9999
        while sector < -1:
            if colors[uidx] < 252:
                sector = colors[uidx]%6
            uidx -= 1
            
        offset = uidx
        
        
        if sector == 0:
            R,G,B = maxon,slider,0
        if sector == 1:
            R,G,B = slider,maxon,0
        if sector == 2:
            R,G,B = 0,maxon,slider
        if sector == 3:
            R,G,B = 0,slider,maxon
        if sector == 4:
            R,G,B = slider,0,maxon
        if sector == 5:
            R,G,B = maxon,0,slider
        Config.append([R,G,B])
        
        theta = (np.pi*(1/3)*sector)+((slider/65536)*np.pi*(1/3))
        U.append(np.cos(theta))
        V.append(np.sin(theta))
        
    return Config,sector,U,V


if Type == 'hypercube':
    
    infile = 'SimulationsHC.txt'

    ShapeC = []
    Node=[]
    sNode = []
    
    readFile = open('%s/HypercubeExt.txt'%outpath,'r')
    sepfile = readFile.read().split('\n')
    for a in range (0,len(sepfile)):
        xandy = sepfile[a].split('\t')
        ShapeC.append([int(xandy[0])-4,int(xandy[1])-4,int(xandy[2])-4])
        Node.append(int(xandy[3]))
        if int(xandy[3])==1:
            sNode.append([int(xandy[0])-4,int(xandy[1])-4,int(xandy[2])-4])
            
if Type == 'sphere' or Type == 'nye':
        
    infile = 'Simulations_Sphere12.txt'
    ShapeC = []
    xxx=[]
    yyy=[]
    zzz=[]
    for a in range (0,12):
        theta = 2*np.pi*(a/12)
        numphi = int(np.abs(round(np.sin(theta)*12)))
        for b in range (0,numphi):
            phi = ((2*np.pi)/numphi)*b
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            ShapeC.append([x,y,z])
            xxx.append(x)
            yyy.append(y)
            zzz.append(z)
    
    sNode = []; idxs=[]
    idx = np.argmin(xxx)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    idx = np.argmax(xxx)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    idx = np.argmin(yyy)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    idx = np.argmax(yyy)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    idx = np.argmin(zzz)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    idx = np.argmax(zzz)
    idxs.append(idx)
    sNode.append(ShapeC[idx])
    
    Node=[]
    for a in range (0,len(ShapeC)):
        if a in idxs:
            Node.append(1)
        else:
            Node.append(0)
            
    #print(np.amin(zzz),np.amax(zzz))
    pollX = [0,0]
    pollY = [0,0]
    pollZ = [-6,0]

if Type == 'pyramid':
    infile = 'SimulationsHC.txt'#temp
    ShapeC = []
    x = [1,2,3,4,5,0.5,0.5,0.5,0.5,0.5,1,2,3,4,5,5.5,5.5,5.5,5.5,5.5,2,3,4,1.5,1.5,1.5,2,3,4,4.5,4.5,4.5,2.5,3.5,3,3]
    y = [0.5,0.5,0.5,0.5,0.5,1,2,3,4,5,5.5,5.5,5.5,5.5,5.5,1,2,3,4,5,1.5,1.5,1.5,2,3,4,4.5,4.5,4.5,2,3,4,3,3,2.5,3.5]
    z = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3]
    z = np.array(z)-0.5
    
    Node = [1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,1]
    
    plusX = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    plusY = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
    plusZ = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#            [1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1]
    
    #cap and floor
    
    for a in range (0,len(x)):
        ShapeC.append([x[a],y[a],z[a]])
    for a in range (0,len(plusX)):
        ShapeC.append([plusX[a],plusY[a],plusZ[a]])
    ShapeC.append([3,3,3])    

    sNode = []
    for a in range (0,len(ShapeC)):
        if Node[a]==1:
            sNode.append(ShapeC[a])
        
    BaseX = [0,6,6,0,0]
    BaseY = [0,0,6,6,0]
    BaseZ = [0,0,0,0,0]
    Beam1x = [0,3,6]
    Beam1y = [0,3,6]
    Beam1z = [0,3,0]
    Beam2x = [6,3,0]
    Beam2y = [0,3,6]
    Beam2z = [0,3,0]

if Type == 'nye':
    targettime = int(sys.argv[3])
    
RadList = []
if Type=='AEM' or Type=='AEMclock':
    infile = 'Simulations_AEM10.txt'
    ShapeC = []
    for a in range (0,10):
        rads = (a/10)*2*np.pi
        RadList.append(rads)
        ShapeC.append([np.cos(rads),np.sin(rads),0])
    sNode = ShapeC
    Node = [1,1,1,1,1,1,1,1,1,1]


SimMI = []    
readFile = open('%s/%s'%(outpath,infile),'r')
sepfile = readFile.read().split('\n')
for a in range (0,len(sepfile)):
    SimMI.append(float(sepfile[a]))
Msorted = sorted(SimMI)


    


Dist=[]
WM = np.zeros((len(sNode),len(sNode)))
ShapeCt = 0
Kct = 0
for a in range (0,len(ShapeC)):
    xm = []
    for b in range (0,len(sNode)):
        dd = (((ShapeC[a][0]-sNode[b][0])**2)+((ShapeC[a][1]-sNode[b][1])**2)+((ShapeC[a][2]-sNode[b][2])**2))**0.5
        if dd>0:
            xm.append(1/(dd**3))
        else:
            xm.append(999)
    if np.amax(xm)==999:#checks if we're on a sNode
        
        for b in range (0,len(sNode)):
            dd = (((ShapeC[a][0]-sNode[b][0])**2)+((ShapeC[a][1]-sNode[b][1])**2)+((ShapeC[a][2]-sNode[b][2])**2))**0.5
            if 1.4<=dd<=4.1:#sqrt of 2 to 4 but within tolerance to allow floating point errors
                WM[ShapeCt,b] = 1
                Kct += 1
        ShapeCt += 1
                
    Dist.append(xm)
    
def RotateAEM(theta):
    for a in range (0,len(ShapeC)):
        ShapeC[a][0] = np.cos(RadList[a]+theta)
        ShapeC[a][1] = np.sin(RadList[a]+theta)
        RadList[a] = RadList[a]+theta
    
def Rotate(theta,axs,doall=False):
    if axs=='x':
        m0 = 2
        m1 = 0
        m2 = 1
    if axs=='y':
        m0 = 0
        m1 = 1
        m2 = 2
    if axs=='z':
        m0 = 1
        m1 = 0
        m2 = 2
    for a in range (0,len(ShapeC)):
        if (-1<=ShapeC[a][m0]<=1 and doall==False) or doall==True:
            r = ((ShapeC[a][m1]**2)+(ShapeC[a][m2]**2))**0.5
            #phi = np.arctan((ShapeC[a][1])/(ShapeC[a][0]))
            phi = math.atan2(ShapeC[a][m2],ShapeC[a][m1])
            phi += theta
            ShapeC[a][m1] = r*np.cos(phi)
            ShapeC[a][m2] = r*np.sin(phi)

def Drop():
    for a in range (0,len(ShapeC)):
        ShapeC[a][2] = ShapeC[a][2]-0.1
            
def SnapInt():
    for a in range (0,len(ShapeC)):
        ShapeC[a][0] = np.rint(ShapeC[a][0])
        ShapeC[a][1] = np.rint(ShapeC[a][1])
        ShapeC[a][2] = np.rint(ShapeC[a][2])



ult_t=[]
Mplt=[]
Mstd=[]
Rplt=[]
Rstd=[]

Xsums=[]
QBsums=[]

ax1y=[]
ax1s=[]
ax1sN=[]  
axQB=[]  
for a in range (0,5):
    Xsums.append([])
    ax1y.append([])

    
    
AimColors = []
Aim_t=[]

AC = Bulk()
AimColors.append(GetColors(AC[0])[0])

AC = Bulk()
AimColors.append(GetColors(AC[0])[0])

Aim_t.append(len(ult_t))

Rot_t=[-999]
RotTyp=[]

MI=[]
MIt=[]

KMlog=[]
CumP=[]


def GetI(uu,vv):
    Ksum=0.0; Kct=0; NCct=0
    for a in range (0,len(uu)):
        for b in range (0,len(uu)):
            if (WM[a,b]==1):#detects neighboring cell, allowing for diagonal directions
                Ksum+= (((uu[a])*(uu[b]))+((vv[a])*(vv[b])))
                Kct+=1
            NCct+=1
            
    SSQ_K = len(uu)
    MORAN = ((len(uu)*Ksum)/(float(Kct)*float(SSQ_K)))
    return MORAN


def animate(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax4t.clear()
    
    reds=[]
    greens=[]
    blues=[]
    
    now_p = time.time()
    now = now_p-(starttime/1000)
    
    
    

    
    if Type=='nye':
        timetodest = targettime - now_p
    else:
        timetodest = None
        
    hrOfD = int((now_p%86400)/3600)
    mnOfD = int((now_p%3600)/60)
    scOfD = int(now_p%60)
    fOfD = str(now_p%1) + '000000'
    microf = fOfD[2:8]
    
    hrRun = int((now)/3600)
    mnRun = int((now%3600)/60)
    scRun = int(now%60)
    
    AC = Bulk()
    t_symbol = AC[2]
    Bird = AC[3]
    
    for a in range (0,len(AC[4])):
        Xsums[a].append(AC[4][a])
    
    
    
    pos = len(ult_t)-Aim_t[-1]
    #print('op enter %d'%pos)
    
    if (np.abs(AC[1]-EX)>ColorThres) and (pos>10):
        aim = GetColors(AC[0])
        
        AimColors.append(aim[0])
        Aim_t.append(len(ult_t))
        #print('spike i %d'%pos)
        
        ival = GetI(aim[2],aim[3])
        tmp_p = M2P(ival)
        pval = 1/tmp_p
        outfile.write('color change,%f\n'%ival)
        MI.append(pval)
        MIt.append((int(now_p*1000)-DayStarted)/3600000)
        KMlog.append(np.log(tmp_p))
        CumP.append(scipy.stats.chi2.sf((np.sum(KMlog)*-2),(2*len(MIt))))
        
        #print(ival)
        
        
        

        
    pos = len(ult_t)-Aim_t[-1]
    #print(pos)
        
    if (pos <= 10):
        Ucolors = []
        for a in range (0,len(sNode)):
            Rnow = AimColors[-1][a][0]
            Rprv = AimColors[-2][a][0]
            Rdo = ((Rnow-Rprv)*(pos/10))+Rprv
    
            Gnow = AimColors[-1][a][1]
            Gprv = AimColors[-2][a][1]
            Gdo = ((Gnow-Gprv)*(pos/10))+Gprv
    
            Bnow = AimColors[-1][a][2]
            Bprv = AimColors[-2][a][2]
            Bdo = ((Bnow-Bprv)*(pos/10))+Bprv
            
            Ucolors.append([Rdo,Gdo,Bdo])
    else:
        Ucolors = AimColors[-1]
        
    Rpos = len(ult_t)-Rot_t[-1]
    if (np.abs(AC[1]-EX)>RotThres) and (Rpos>10):
        Rot_t.append(len(ult_t))
        RotTyp.append(GetColors(AC[0])[1])
        outfile.write('rotation\n')
    Rpos = len(ult_t)-Rot_t[-1]
    if Rpos < 10 and (Type=='hypercube'):
        if Type=='hypercube':
            allbool = False
        else:
            
            allbool = True
        if RotTyp[-1]==0:
            Rotate(np.pi/20,'x',doall=allbool)
        if RotTyp[-1]==1:
            Rotate(-np.pi/20,'x',doall=allbool)
        if RotTyp[-1]==2:
            Rotate(np.pi/20,'y',doall=allbool)
        if RotTyp[-1]==3:
            Rotate(-np.pi/20,'y',doall=allbool)
        if RotTyp[-1]==4:
            Rotate(np.pi/20,'z',doall=allbool)
        if RotTyp[-1]==5:
            Rotate(-np.pi/20,'z',doall=allbool)
        if Rpos==9 and Type=='hypercube':
            SnapInt()
    if Rpos < 10 and (Type=='AEMclock' or Type=='AEM'):
        if 0<=RotTyp[-1]<=2:
            RotateAEM(np.pi/30)
        else:
            RotateAEM(np.pi/-30)

    
    if Type =='nye' and 0<=timetodest<=60:
        Drop()
        
    NowXT = (int(now_p*1000)-DayStarted)/3600000
    ult_t.append(NowXT)
    NodeCt=0
    for a in range (0,len(ShapeC)):
        if Node[a]==1:
            red = Ucolors[NodeCt][0]/(256**2)
            green = Ucolors[NodeCt][1]/(256**2)
            blue = Ucolors[NodeCt][2]/(256**2)
            NodeCt += 1
            reds.append(red)
            greens.append(green)
            blues.append(blue)
            ax1.scatter(ShapeC[a][0],ShapeC[a][1],ShapeC[a][2],color=[red,green,blue],edgecolors=None,s=DotSize)
    for a in range (0,len(ShapeC)):
        if Node[a]==0:
            red = np.average(reds,weights=Dist[a])
            green = np.average(greens,weights=Dist[a])
            blue = np.average(blues,weights=Dist[a])
            ax1.scatter(ShapeC[a][0],ShapeC[a][1],ShapeC[a][2],color=[red,green,blue],edgecolors=None,s=DotSize)
            
            
    M = len(AimColors)-2
    if pos <= 10:
        Nt = (len(ult_t)-(10*M)+(10-pos))-11
    else:
        Nt = (len(ult_t)-(10*M))-11
        
    R = len(RotTyp)
    if Rpos <= 10:
        NtR = (len(ult_t)-(10*R)+(10-Rpos))-11
    else:
        NtR = (len(ult_t)-(10*R))-11
    
    Mplt.append(M - (Pmod_Color*Nt))
    Mstd.append(((Nt*Pmod_Color*(1-Pmod_Color))**0.5)*1.65)
    
    Rplt.append(R - (Pmod_Rot*NtR))
    Rstd.append(((NtR*Pmod_Rot*(1-Pmod_Rot))**0.5)*1.65)
            
    if Type == 'AEMclock':
        plt.suptitle('%02d:%02d:%02d.%s%s UTC      |      T+ %02d:%02d:%02d'%(hrOfD,mnOfD,scOfD,microf,t_symbol,hrRun,mnRun,scRun),color=[np.average(reds),np.average(greens),np.average(blues)],size=wordsize)
        ax1.text(0,0,0,"%s"%(Bird),ha='center',va='center',color=[np.average(reds),np.average(greens),np.average(blues)],size=wordsize)
        
        
        
    ax1.grid(False)
    ax1.set_axis_off()
    if Type=='nye':
        ax1.plot(pollX,pollY,pollZ,color='white')
        ax1.set_xlim3d(-3.5,3.5)
        ax1.set_ylim3d(-3.5,3.5)
        ax1.set_zlim3d(-6.5,0.5)
        hoursleft = int(timetodest/3600)
        minleft = int(timetodest/60)%60
        secleft = int(timetodest%60)
        #plt.suptitle('%02d:%02d:%02d'%(hoursleft,minleft,secleft),size=70)
        Rotate(np.pi/20,'x',doall=True)
        
    
    
    

    for a in range (0,len(riprise)):
        if StartXT<=riprise[a]<=NowXT:
            ax2.axvline(x=riprise[a],color='lightblue')
            ax3.axvline(x=riprise[a],color='lightblue')
    for a in range (0,len(ripset)):
        if StartXT<=ripset[a]<=NowXT:
            ax2.axvline(x=ripset[a],color='pink')
            ax3.axvline(x=ripset[a],color='pink')
            
    QBsums.append(AC[1])
    axQB.append(np.sum(QBsums)-(len(ult_t)*NEDspeed*8*0.5))
    for a in range (0,len(Xsums)):
        ax1y[a].append(np.sum(Xsums[a])-(len(ult_t)*NEDspeed*8*0.5))
        #print(a,(np.sum(Xsums[a])-(len(ult_t)*NEDspeed*8*0.5)))
    ax1s.append(((len(ult_t)*NEDspeed*8*0.25)**0.5)*1.96)
    ax1sN.append(((len(ult_t)*NEDspeed*8*0.25)**0.5)*-1.96)
    if TurboUse==True:
        ax2.plot(ult_t,ax1y[0],color='red',linewidth='1')
        for a in range (1,len(ax1y)):
            ax2.plot(ult_t,ax1y[a],color='lightgray',linewidth='1')
    else:
        for a in range (0,len(ax1y)-1):
            ax2.plot(ult_t,ax1y[a],color='lightgray',linewidth='1')
    ax2.plot(ult_t,axQB,color='magenta',linewidth='1')
    
    ax2.plot(ult_t,ax1s,color='aqua',linestyle='--')
    ax2.plot(ult_t,ax1sN,color='aqua',linestyle='--')      
    
    ax3.plot(ult_t,Mplt,color='aqua')
    ax3.plot(ult_t,Mstd,color='aqua',linestyle='--')
    ax3.plot(ult_t,Rplt,color='red')
    ax3.plot(ult_t,Rstd,color='red',linestyle='--')
    ax4.plot(MIt,MI)
    ax4.set_yscale('log')
    
    
    ax4t.plot(MIt,CumP,color='red')
    ax4t.set_ylim([0,1])
    
    ax2.set_ylabel('raw')
    ax3.set_ylabel('Rotation / Color Change')
    ax4.set_ylabel('Color Coherence 1/p')
    
    if zoomsto[0]==0:
        ll= ult_t[-1]-(1/60)
        ul=ult_t[-1]
        ax2.set_xlim([ll,ul])
        ax3.set_xlim([ll,ul])
        ax4.set_xlim([ll,ul])
    
    
    
    if Type == 'pyramid':
        ax1.plot(BaseX,BaseY,BaseZ,color='white')
        ax1.plot(Beam1x,Beam1y,Beam1z,color='white')
        ax1.plot(Beam2x,Beam2y,Beam2z,color='white')
        
    
    #if len(ult_t)%FileMax==0:
    #    outfile.close()
    #    fnum = len(ult_t)/FileMax
    #    outfile.open('%s/HYPERCUBE_%s_%d_%d.txt'%(outpath,Type,starttime,fnum),'w')
        
                
ani = animation.FuncAnimation(fig, animate, interval=1000)

plt.show()


#similar rgb algorithm to aurora, so copy. and MI after approval and have next step.   rotate same thing.   for now just black background... and blend nodes.
#px against chance ... loadshape function   px works any shape.   at least MI for hypercube can work out and px against chance on right side.   inc A0. start simple.  see if works better with A0. mk pretty as possible.
#MI, rotate, pretty, A0 later., color gliding    1 p value per cube and then the whole thing.
#imagine just outside of hypercube
#a0...but for now grab ea min   A0threshold for new colors and A0 threshold for rotate any.  MI.  mk more.

"""
n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

"""