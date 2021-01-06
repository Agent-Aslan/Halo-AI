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
warnings.simplefilter('ignore')

#stats = rotateZmod, colorZmod, MI/grid, MI total. can we shrink plot area so 4 params in top to btm and whole left side is HC?/pyramid/whatever?
#not doing max per thousand here.

NEDspeed = 1000
ColorZ = 1.65
RotZ = 1.85
outpath = 'C:/Users/Aslan/HALO/DATA'
Type = 'PMD'

Type = sys.argv[1]

###########


starttime = int(time.time()*1000)
outfile = open('%s/HYPERCUBE_%d.txt'%(outpath,starttime),'w')
outfile.write('ColorZ: %f RotZ: %f\n'%(ColorZ,RotZ))

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




ports=dict()  
ports_avaiable = list(list_ports.comports())
rng_com_port = None
for temp in ports_avaiable:
    #if temp[1].startswith("TrueRNG"):
    if "pro" in temp[1]:
        print('Found:           ' + str(temp))
        if rng_com_port == None:        # always chooses the 1st TrueRNG found
            rng_com_port=str(temp[0])
print('Using com port:  ' + str(rng_com_port))
print('==================================================')
sys.stdout.flush()
try:
    ser = serial.Serial(port=rng_com_port,timeout=10)  # timeout set at 10 seconds in case the read fails
except:
    print('Port Not Usable!')
    print('Do you have permissions set to read ' + rng_com_port + ' ?')
if(ser.isOpen() == False):
    ser.open()
ser.setDTR(True)
ser.flushInput()
sys.stdout.flush()








def Bulk():
    ser.flushInput()
    x = ser.read(NEDspeed)
    bitct = 0
    for a in range (0,len(x)):
        outfile.write('%d,'%x[a])
        strnode = str(bin(256+int(x[a])))[3:]
        bitct += int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7])        
        
    outfile.write('%d\n'%(int(time.time()*1000)))
    outfile.flush()
    os.fsync(outfile.fileno())
    return x,bitct

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

    ShapeC = []
    Node=[]
    sNode = []
    
    readFile = open('C:/Users/Aslan/HALO/HypercubeExt.txt','r')
    sepfile = readFile.read().split('\n')
    for a in range (0,len(sepfile)):
        xandy = sepfile[a].split('\t')
        ShapeC.append([int(xandy[0])-4,int(xandy[1])-4,int(xandy[2])-4])
        Node.append(int(xandy[3]))
        if int(xandy[3])==1:
            sNode.append([int(xandy[0])-4,int(xandy[1])-4,int(xandy[2])-4])
            
if Type == 'sphere':
        
        
    ShapeC = []
    xxx=[]
    yyy=[]
    zzz=[]
    for a in range (0,15):
        theta = 2*np.pi*(a/15)
        numphi = int(np.abs(round(np.sin(theta)*15)))
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

if Type == 'pyramid':
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
    
def Rotate(theta,axs):
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
        if -1<=ShapeC[a][m0]<=1:
            r = ((ShapeC[a][m1]**2)+(ShapeC[a][m2]**2))**0.5
            #phi = np.arctan((ShapeC[a][1])/(ShapeC[a][0]))
            phi = math.atan2(ShapeC[a][m2],ShapeC[a][m1])
            phi += theta
            ShapeC[a][m1] = r*np.cos(phi)
            ShapeC[a][m2] = r*np.sin(phi)
            
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
    
    reds=[]
    greens=[]
    blues=[]
    
    now = time.time()-(starttime/1000)
    
    AC = Bulk()
    
    
    pos = len(ult_t)-Aim_t[-1]
    #print('op enter %d'%pos)
    
    if (np.abs(AC[1]-EX)>ColorThres) and (pos>10):
        aim = GetColors(AC[0])
        AimColors.append(aim[0])
        Aim_t.append(len(ult_t))
        #print('spike i %d'%pos)
        
        ival = GetI(aim[2],aim[3])
        
        MI.append(ival)
        MIt.append(now)
        
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
    Rpos = len(ult_t)-Rot_t[-1]
    if Rpos < 10 and Type=='hypercube':
        if RotTyp[-1]==0:
            Rotate(np.pi/20,'x')
        if RotTyp[-1]==1:
            Rotate(-np.pi/20,'x')
        if RotTyp[-1]==2:
            Rotate(np.pi/20,'y')
        if RotTyp[-1]==3:
            Rotate(-np.pi/20,'y')
        if RotTyp[-1]==4:
            Rotate(np.pi/20,'z')
        if RotTyp[-1]==5:
            Rotate(-np.pi/20,'z')
        if Rpos==9:
            SnapInt()
    
    
    ult_t.append(now)
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
            ax1.scatter(ShapeC[a][0],ShapeC[a][1],ShapeC[a][2],color=[red,green,blue],edgecolors=None,s=500)
    for a in range (0,len(ShapeC)):
        if Node[a]==0:
            red = np.average(reds,weights=Dist[a])
            green = np.average(greens,weights=Dist[a])
            blue = np.average(blues,weights=Dist[a])
            ax1.scatter(ShapeC[a][0],ShapeC[a][1],ShapeC[a][2],color=[red,green,blue],edgecolors=None,s=500)
            
            
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
            
    ax1.grid(False)
    ax1.set_axis_off()
    ax2.plot(ult_t,Mplt)
    ax2.plot(ult_t,Mstd)
    ax3.plot(ult_t,Rplt)
    ax3.plot(ult_t,Rstd)
    ax4.plot(MIt,MI)
    
    ax2.set_ylabel('Color Change')
    ax3.set_ylabel('Rotation')
    ax4.set_ylabel('Correlation coeff')
    
    if Type == 'pyramid':
        ax1.plot(BaseX,BaseY,BaseZ,color='white')
        ax1.plot(Beam1x,Beam1y,Beam1z,color='white')
        ax1.plot(Beam2x,Beam2y,Beam2z,color='white')
        

                
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