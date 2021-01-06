import time
import fluidsynth
import numpy
import os
import sys
import serial
from serial.tools import list_ports
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.animation as animation
from PIL import Image
from PIL import ImageOps

#If running on HALO, must run from anaconda prompt. Animation will not work in spyder IDE!

UseTrueRNG = False
ChordSound = 101
ChordVol = 100
PentSound = 79#104
PentVol = 100
Key = 48#starting key, 60 = middle C
Kmax = 70
RNG_Interval = 0.2
RNG_BytesPerInterval = 25
ChordThres = 25
PentThres = 13#must be <ChordThres
OutPath = 'C:/Users/Aslan/Documents/HALO Development/MeditationRNG'

# im = Image.open('C:/Users/Daniel/Downloads/dream-image.jpg')
Pmod = scipy.stats.norm.sf(ChordThres/((RNG_BytesPerInterval*2)**0.5))*2

def Modulate():
    
    K = KeyList[-1]
    
    fs.noteoff(0, K)
    fs.noteoff(0, K+4)
    fs.noteoff(0, K+7)
    fs.noteoff(0, K+12)
    
    if (K>Kmax):
        K+= -7
    else:
        K+=5
        
    fs.noteon(0, K, ChordVol)
    fs.noteon(0, K+4, ChordVol)
    fs.noteon(0, K+7, ChordVol)
    fs.noteon(0, K+12, ChordVol)
        
    #NewPent = [K-12,K-10,K-8,K-5,K-3,K,K+2,K+4,K+7,K+9,K+12]
    
    KeyList.append(K)
    
    #return NewPent


totaltime=[]
totalmods=[]
P1=[]

def animate(i):
    
    KK = KeyList[-1]
    Pent = [KK-12,KK-10,KK-8,KK-5,KK-3,KK,KK+2,KK+4,KK+7,KK+9,KK+12]
    
    current_time = time.time()
    totaltime.append(current_time-(starttime/1000.0))
    
    
    if (UseTrueRNG==True):
        ser.flushInput()
        x = ser.read(RNG_BytesPerInterval)
    else:
        x = numpy.random.randint(0,256,RNG_BytesPerInterval)
    rc=0
    for a in range (0,RNG_BytesPerInterval):
        rc+=lib[x[a]]
        outfile.write('%d,'%(x[a]))
    outfile.write('%d\n'%(int(current_time*1000)))
    if (Pul<=rc<Cul) or (Cll<rc<=Pll):
        if (len(Gigasavenote)==0):
            fs1.noteoff(0, 0)
        else:
            fs1.noteoff(0, Gigasavenote[-1])
        note = int(x[0]/23.1819)
        Gigasavenote.append(Pent[note])
        fs1.noteon(0, Pent[note], PentVol)
    #time.sleep(RNG_Interval)
    #fs1.noteoff(0, Pent[note])
    #switch = numpy.random.randint(0,20)
    if (rc>=Cul) or (rc<=Cll):
        Modulate()
        fs1.noteoff(0, 0)
        outfile.write('modulation\n')
        
    #print(current_time)
    
    EX = len(totaltime)*Pmod
    totalmods.append((len(KeyList)-2)-EX)
    P1.append(((len(totaltime)*Pmod*(1-Pmod))**0.5)*1.65)

    MinY = numpy.amin([numpy.amin(P1),numpy.amin(totalmods)])
    MaxY = numpy.amax([numpy.amax(P1),numpy.amax(totalmods)])
    MaxX = totaltime[-1]+1

    ax1.clear()
    # ax1.imshow(im, aspect='auto', extent=(0,MaxX,MinY,MaxY))
    ax1.set_xlim(0,MaxX)
    ax1.set_ylim(MinY,MaxY)
    ax1.plot(totaltime,totalmods)
    ax1.plot(totaltime,P1)
    #print('ok')
Gigasavenote=[]





starttime = int(time.time()*1000)
outfile = open('%s\RNG_%d.txt'%(OutPath,starttime),'w')
outfile.write('%s,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d\n'%(UseTrueRNG,ChordSound,ChordVol,PentSound,PentVol,Key,Kmax,RNG_Interval,RNG_BytesPerInterval,ChordThres,PentThres))


lib=[]
readFile = open('%s\Conversion.txt'%(OutPath), 'r')
sepfile = readFile.read().split('\n')
for b in range (0,len(sepfile)):
    xandy = sepfile[b].split('\t')
    lib.append(float(xandy[0]))


if (UseTrueRNG==True):
    ports=dict()  
    ports_avaiable = list(list_ports.comports())
    rng_com_port = None
    for temp in ports_avaiable:
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


fs = fluidsynth.Synth()
fs.start(driver = 'dsound')  # use DirectSound driver
sfid = fs.sfload(r'C:\Users\Aslan\Documents\HALO Development\FluidTest\FluidR3_GM\FluidR3_GM.sf2')  # replace path as needed


fs1 = fluidsynth.Synth()
fs1.start(driver = 'dsound')  # use DirectSound driver
sfid1 = fs1.sfload(r'C:\Users\Aslan\Documents\HALO Development\FluidTest\FluidR3_GM\FluidR3_GM.sf2')  # replace path as needed
fs1.program_select(0, sfid, 0, PentSound)

fs.program_select(0, sfid, 0, ChordSound)

Pll = (RNG_BytesPerInterval*4)-PentThres
Pul = (RNG_BytesPerInterval*4)+PentThres
Cll = (RNG_BytesPerInterval*4)-ChordThres
Cul = (RNG_BytesPerInterval*4)+ChordThres



KeyList = [Key-5]

fig = plt.figure()
ax1 = fig.add_subplot(111)



savenote = 0
#Pent = [K-12,K-10,K-8,K-5,K-3,K,K+2,K+4,K+7,K+9,K+12]


#fs.noteon(60, 60, ChordVol)
    
Modulate()
print('ok')
ani = animation.FuncAnimation(fig, animate, interval=int(RNG_Interval*1000))
plt.show()


"""
Play = True
while Play==True:
    


    fs.program_select(0, sfid, 0, ChordSound)
    fs.noteon(0, K, ChordVol)
    fs.noteon(0, K+4, ChordVol)
    fs.noteon(0, K+7, ChordVol)
    fs.noteon(0, K+12, ChordVol)
    
    Pent = [K-12,K-10,K-8,K-5,K-3,K,K+2,K+4,K+7,K+9,K+12]
    
    StayOnKey = True
    while StayOnKey==True:
        if (UseTrueRNG==True):
            ser.flushInput()
            x = ser.read(RNG_BytesPerInterval)
        else:
            x = numpy.random.randint(0,256,RNG_BytesPerInterval)
        rc=0
        for a in range (0,RNG_BytesPerInterval):
            rc+=lib[x[a]]
            outfile.write('%d,'%(x[a]))
        outfile.write('%d\n'%(int(time.time()*1000)))
        if (Pul<=rc<Cul) or (Cll<rc<=Pll):
            fs1.noteoff(0, savenote)
            note = int(x[0]/23.1819)
            savenote = Pent[note]
            fs1.noteon(0, Pent[note], PentVol)
        time.sleep(RNG_Interval)
        #fs1.noteoff(0, Pent[note])
        #switch = numpy.random.randint(0,20)
        if (rc>=Cul) or (rc<=Cll):
            StayOnKey = False
            fs1.noteoff(0, savenote)
            outfile.write('modulation\n')

            
    fs.noteoff(0, K)
    fs.noteoff(0, K+4)
    fs.noteoff(0, K+7)
    fs.noteoff(0, K+12)
    
    if (K>Kmax):
        K+= -7
    else:
        K+=5
        
"""

#fs1.noteon(0, 60, 127)
#fs1.noteon(0, 67, 127)
#fs1.noteon(0, 76, 127)

#time.sleep(3.0)

#fs.delete()