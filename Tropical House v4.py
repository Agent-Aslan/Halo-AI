import time
import fluidsynth
import numpy
from playsound import playsound
#import threading
import concurrent.futures
#import pygame
import sys
import serial
from serial.tools import list_ports
import matplotlib.pyplot as plt

import statsmodels.api as sm
import scipy.signal
import scipy.fftpack

import pylab
from pylab import *
from numpy import ma
from pylab import polyfit
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import os
import matplotlib.cm as cm

#import multiprocessing

Scale = [67, 69, 72, 74, 76, 79, 81]
SoloScale = [60,62,64,67,69,72,74,75,76,79,81,84]
ScaleRoot = 72
SoloAccentIdx = [0,3,6,9,12]


RNGfreq = 200
PluckSpd = 0.02
NoteSpeed = 0.13
ChordPluck = False
#MTrials = 1000
FreeRhythm = False
UseTrueRNG = True
OutPath = 'C:/Users/Aslan/Documents/TropicalHouseRNG'
ThresholdZ = 1.85
SoloZ = 0.5#1.3
ModAmount = 2

UnivFreq = 1.0/(PluckSpd+NoteSpeed)

starttime = int(time.time()*1000)
print(starttime)
outfile = open('%s\output\RNG_%d.txt'%(OutPath,starttime),'w')

if UseTrueRNG == True:
    # Create ports variable as dictionary
    ports=dict()  
    
    # Call list_ports to get com port info 
    ports_avaiable = list(list_ports.comports())
    
    # Set default of None for com port
    rng_com_port = None
    
    # Loop on all available ports to find TrueRNG
    for temp in ports_avaiable:
        if temp[1].startswith("TrueRNG"):
            #print('Found:           ' + str(temp))
            if rng_com_port == None:        # always chooses the 1st TrueRNG found
                rng_com_port=str(temp[0])
    
    # Print which port we're using
    #print('Using com port:  ' + str(rng_com_port))
    
    # Print block size and number of loops
    #print('==================================================')
    sys.stdout.flush()
    
    # Try to setup and open the comport
    ser = serial.Serial(port=rng_com_port,timeout=10)  # timeout set at 10 seconds in case the read fails
        
    # Open the serial port if it isn't open
    if(ser.isOpen() == False):
        ser.open()
    
    # Set Data Terminal Ready to start flow
    ser.setDTR(True)   
    
    # This clears the receive buffer so we aren't using buffered data
    ser.flushInput()  




fs = fluidsynth.Synth()
fs.start(driver = 'dsound')  # use DirectSound driver
sfid0 = fs.sfload(r'C:\Users\Aslan\Documents\TropicalHouseRNG\soundfonts\Timbres Of Heaven (XGM) 3.94.sf2')  # replace path as needed

fs1 = fluidsynth.Synth()
fs1.start(driver = 'dsound')  # use DirectSound driver
sfid1 = fs1.sfload(r'C:\Users\Aslan\Documents\TropicalHouseRNG\soundfonts\Timbres Of Heaven (XGM) 3.94.sf2')  # replace path as needed

fs2 = fluidsynth.Synth()
fs2.start(driver = 'dsound')  # use DirectSound driver
sfid2 = fs2.sfload(r'C:\Users\Aslan\Documents\TropicalHouseRNG\soundfonts\Timbres Of Heaven (XGM) 3.94.sf2')  # replace path as needed

fs3 = fluidsynth.Synth()
fs3.start(driver = 'dsound')  # use DirectSound driver
sfid3 = fs3.sfload(r'C:\Users\Aslan\Documents\TropicalHouseRNG\soundfonts\Timbres Of Heaven (XGM) 3.94.sf2')  # replace path as needed

fs4 = fluidsynth.Synth()
fs4.start(driver = 'dsound')  # use DirectSound driver
sfid4 = fs4.sfload(r'C:\Users\Aslan\Documents\TropicalHouseRNG\soundfonts\Timbres Of Heaven (XGM) 3.94.sf2')  # replace path as needed

#NoteSeq = [60,62,64,65,67,65,64,62,60]
#


lib=[]
readFile = open('%s\Conversion.txt'%(OutPath), 'r')
sepfile = readFile.read().split('\n')
for b in range (0,len(sepfile)):
    xandy = sepfile[b].split('\t')
    lib.append(float(xandy[0]))
    
    
#getting around this for now by threading io multiprocess
def RNG_Bulk():
    bulk=[]
    ser.flushInput()
    
    #ser.flushInput()
    rc=0
    for a in range (0,RNGfreq):
        x = ser.read(64)
        for b in range (0,64):
            rc+=lib[x[b]]
            bulk.append(x[b])
            outfile.write('%d,'%(x[b]))
        outfile.write('\n')
    outfile.write('%d\n'%(int(time.time()*1000)))
    return rc,bulk
        

def PlayClap():
    playsound('C:/Users/Aslan/Documents/TropicalHouseRNG/soundfonts/snap3.wav')
    #print('clap')
    #pygame.mixer.Sound.play(clap_sound)
    
def PlayKick():
    playsound('C:/Users/Aslan/Documents/TropicalHouseRNG/soundfonts/kick5.wav')
    #print('kick')

def PlayHat():
    playsound('C:/Users/Aslan/Documents/TropicalHouseRNG/soundfonts/hat1.wav')
    #print('hat')
    
def ChordID(Cstr,octave,inversion):
    if Cstr.startswith('A'):
        idx=57
    if Cstr.startswith('B'):
        idx=59
    if Cstr.startswith('C'):
        idx=60
    if Cstr.startswith('D'):
        idx=62
    if Cstr.startswith('E'):
        idx=64
    if Cstr.startswith('F'):
        idx=65
    if Cstr.startswith('G'):
        idx=67
        
    if '#' in Cstr:
        idx+= 1
    if 'b' in Cstr:
        idx+= -1
        
    idx += (octave*12)
    
    if 'm' in Cstr:
        Carr = [idx,idx+3,idx+7]
    else:
        Carr = [idx,idx+4,idx+7]
        
    if inversion==2:
        Carr[0] += 12
    if inversion==3:
        Carr[0] += 12
        Carr[1] += 12
    
    return Carr
    
def AbsDist(note_arr):#just pull in scale as arg if not work
    x_use=[]
    u_scale = numpy.array(Scale)
    for a in range (0,len(note_arr)):
        if (note_arr[a]>0):
            x_use.append(numpy.where(u_scale==note_arr[a])[0][0])
    x_arr=[]
    RepeatCt=0
    for a in range (1,len(x_use)):
        xx_use = numpy.abs(x_use[a]-x_use[a-1])
        x_arr.append(xx_use)
        if (xx_use==0):
            RepeatCt+=1
    return numpy.nanmean(x_arr),RepeatCt/(len(x_use)-1)
    
def MaxJump(NoteSeqS,NoteDurS):
    x_jump=[]
    for a in range (1,len(NoteSeqS)):
        x_jump.append((numpy.abs(NoteSeqS[a]-NoteSeqS[a-1]))/NoteDurS[a-1])
    return numpy.nanmean(x_jump)
        

def FFTdist(NoteSeqS,NoteDurS,Hz,det):
    FFT_seq=[]
    for a in range (0,len(NoteSeqS)):#helps that 64 IS a valid multiple...not sure how to construct
        for b in range (0,NoteDurS[a]):
            if (NoteSeqS[a]==-1):
                FFT_seq.append(ScaleRoot)
            else:
                FFT_seq.append(NoteSeqS[a])
            #FFT_seq.append(NoteSeqS[a])
            
    FFT_seq = FFT_seq#*64 # streatches length...so no trend
    if det==True:
        FFT_det=scipy.signal.detrend(FFT_seq)
    else:
        FFT_det = FFT_seq
    
    U_fft = scipy.fftpack.rfft(FFT_det)
    U_psd = U_fft * U_fft.conjugate()
        
    N = len(FFT_det)
    fs = Hz#Hz
    df = fs/float(N)
    freq = numpy.arange(0,fs,df)
    Nfi = numpy.intc(numpy.floor(N/2))
    testfreq =  freq[1:Nfi+1]
    xU_psd = U_psd/df/N**2
    sU_psd = numpy.real(xU_psd[1:Nfi+1])
    
    Log_testfreq = numpy.log10(testfreq)
    Log_sU_psd = numpy.log10(sU_psd)
    
    m1,b1 = numpy.polyfit(Log_testfreq,Log_sU_psd,1)
    return m1*-1



def GetMelody(Sc,NumNotes,rnum,rests=None,C_lock=None,M_lock=None):
    #print(Sc,NumNotes,rnum,'\n')
    buildM=[]
    for a in range (0,NumNotes):
        if rests:
            if a in rests:
                buildM.append(-1)
            else:
                buildM.append(Sc[rnum[a]%len(Sc)])
        else:
            buildM.append(Sc[rnum[a]%len(Sc)])
    if C_lock:
        for a in range (0,len(C_lock)):
            idx = C_lock[a][0]
            ModSc = C_lock[a][1:]
            #buildM[idx] = ModSc[numpy.random.randint(0,len(ModSc))]
            buildM[idx] = ModSc[rnum[idx]%len(ModSc)]
    if M_lock:
        for a in range (0,len(M_lock)):
            ll = M_lock[a][0]
            ul = M_lock[a][1]
            paste = M_lock[a][2:]
            base = buildM[ll:ul]#remember the +1 aspect when putting this in
            for b in range (0,len(paste)):
                dx = paste[b]
                for c in range (0,len(base)):
                    buildM[dx+c] = base[c]
    return buildM
    
    
def GetNotes(Progression,MTrials,numbers):

    if Progression==1:
        PxCloud = [1,1,4,1,4,3,2,4,1,2,1,3,3,0,4,0]#/4
        #RtmPre = numpy.random.randint(0,4,16)
        RtmPre = numpy.array(numbers[0:16])%4
        Rtm=[]
        for a in range (0,len(PxCloud)):
            if RtmPre[a]<PxCloud[a]:
                Rtm.append(1)
            else:
                Rtm.append(0)
        for a in range (0,16):
            Rtm.append(Rtm[a])
        PxCloud = [1,1,4,1,4,3,2,4,1,3,1,3,3,1,3,0,2,1,4,2,4,4,1,4,0,3,3,1,4,2,4,2]#/4
        #RtmPre = numpy.random.randint(0,4,32)
        RtmPre = numpy.array(numbers[16:48])%4
        for a in range (0,len(PxCloud)):
            if RtmPre[a]<PxCloud[a]:
                Rtm.append(1)
            else:
                Rtm.append(0)
        pNoteDur = []
        InRests = None
        if Rtm[0]==0:
            InRests = [0]
            LenR = 0; found = 0
            for a in range (0,len(Rtm)):
                if (Rtm[a]==0) and (found==0):
                    LenR +=1
                else:
                    found +=1
            pNoteDur.append(LenR)
        for a in range (0,len(Rtm)):
            if Rtm[a]==1:
                LenR = 1; found = 0
                if (a<len(Rtm)-1):
                    for b in range (a+1,len(Rtm)):
                        if (Rtm[b]==0) and (found==0):
                            LenR +=1
                        else:
                            found +=1
                pNoteDur.append(LenR)
        MelLock=[[]]
        CumDur=0
        CpyList=[]
        found=0
        for a in range(0,len(pNoteDur)):
            if CumDur<8:
                CpyList.append(a)
            if CumDur>=16 and found==0:
                PastePt = a
                found += 1
            CumDur+=pNoteDur[a]
        if InRests:
            MelLock[0].append(1)
        else:
            MelLock[0].append(0)
        #MelLock[0].append(numpy.amin(CpyList))
        MelLock[0].append(numpy.amax(CpyList)+1)
        MelLock[0].append(PastePt)
        ChLock = [[],[],[],[]]
        if InRests:
            ChLock[0].append(1)
        else:
            ChLock[0].append(0)
        ChLock[0].append(69)
        ChLock[0].append(72)
        ChLock[0].append(81)
        CumDur=0
        found=0
        for a in range (0,len(pNoteDur)):
            if CumDur>=16 and found==0:
                ChLock[1].append(a)
                ChLock[1].append(69)
                ChLock[1].append(72)
                ChLock[1].append(81)
                found += 1
            CumDur += pNoteDur[a]
        CumDur=0
        found=0
        for a in range (0,len(pNoteDur)):
            if CumDur>=32 and found==0:
                ChLock[2].append(a)
                ChLock[2].append(72)
                ChLock[2].append(76)
                ChLock[2].append(79)
                found += 1
            CumDur += pNoteDur[a]
        ChLock[3].append(len(pNoteDur)-1)
        ChLock[3].append(72)
        
    
        
    if Progression==2:
    
        PxCloud = [1,1,4,1,4,3,2,4,1,2,1,3,3,0,4,0]#/4
        #RtmPre = numpy.random.randint(0,4,16)
        RtmPre = numpy.array(numbers[0:16])%4
        Rtm=[]
        for a in range (0,len(PxCloud)):
            if RtmPre[a]<PxCloud[a]:
                Rtm.append(1)
            else:
                Rtm.append(0)
        for a in range (0,16):
            Rtm.append(Rtm[a])
        PxCloud = [1,1,4,1,4,3,2,4,1,3,1,3,3,1,3,0,2,1,4,2,4,4,1,4,0,3,3,1,4,2,4,2]#/4
        #RtmPre = numpy.random.randint(0,4,32)
        RtmPre = numpy.array(numbers[16:48])%4
        for a in range (0,len(PxCloud)):
            if RtmPre[a]<PxCloud[a]:
                Rtm.append(1)
            else:
                Rtm.append(0)
        pNoteDur = []
        InRests = None
        if Rtm[0]==0:
            InRests = [0]
            LenR = 0; found = 0
            for a in range (0,len(Rtm)):
                if (Rtm[a]==0) and (found==0):
                    LenR +=1
                else:
                    found +=1
            pNoteDur.append(LenR)
        for a in range (0,len(Rtm)):
            if Rtm[a]==1:
                LenR = 1; found = 0
                if (a<len(Rtm)-1):
                    for b in range (a+1,len(Rtm)):
                        if (Rtm[b]==0) and (found==0):
                            LenR +=1
                        else:
                            found +=1
                pNoteDur.append(LenR)
        MelLock=[[]]
        CumDur=0
        CpyList=[]
        found=0
        for a in range(0,len(pNoteDur)):
            if CumDur<8:
                CpyList.append(a)
            if CumDur>=16 and found==0:
                PastePt = a
                found += 1
            CumDur+=pNoteDur[a]
        if InRests:
            MelLock[0].append(1)
        else:
            MelLock[0].append(0)
        #MelLock[0].append(numpy.amin(CpyList))
        MelLock[0].append(numpy.amax(CpyList)+1)
        MelLock[0].append(PastePt)
        ChLock = [[],[],[],[]]
        if InRests:
            ChLock[0].append(1)
        else:
            ChLock[0].append(0)
        ChLock[0].append(72)
        ChLock[0].append(74)
        ChLock[0].append(79)
        CumDur=0
        found=0
        for a in range (0,len(pNoteDur)):
            if CumDur>=16 and found==0:
                ChLock[1].append(a)
                ChLock[1].append(72)
                ChLock[1].append(74)
                ChLock[1].append(79)
                found += 1
            CumDur += pNoteDur[a]
        CumDur=0
        found=0
        for a in range (0,len(pNoteDur)):
            if CumDur>=32 and found==0:
                ChLock[2].append(a)
                ChLock[2].append(69)
                ChLock[2].append(72)
                ChLock[2].append(81)
                found += 1
            CumDur += pNoteDur[a]
        ChLock[3].append(len(pNoteDur)-1)
        ChLock[3].append(72)
        
    if Progression==3:
        #Rtm = numpy.random.randint(0,2,64)
        Rtm = numpy.array(numbers[0:64])%2
        pNoteDur = []
        InRests = None
        if Rtm[0]==0:
            InRests = [0]
            LenR = 0; found = 0
            for a in range (0,len(Rtm)):
                if (Rtm[a]==0) and (found==0):
                    LenR +=1
                else:
                    found +=1
            pNoteDur.append(LenR)
        for a in range (0,len(Rtm)):
            if Rtm[a]==1:
                LenR = 1; found = 0
                if (a<len(Rtm)-1):
                    for b in range (a+1,len(Rtm)):
                        if (Rtm[b]==0) and (found==0):
                            LenR +=1
                        else:
                            found +=1
                pNoteDur.append(LenR)
    if Progression==4:
        pNoteDur = [2,2,1,1,1,1,3,3,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,2,3,2,2,1,1,1,3,2,2,2]
    
    
    Mx_save = 999999
    
    save_Dist=[]; save_Rep=[]
    ChangeCdtn=0
    found=0
    
    for mm in range (0,MTrials):
        #if FreeRhythm==True:
        #    notes_save = GetMelody(Scale,len(pNoteDur),InRests)
        #else:
        #    notes_save = GetMelody(Scale,38,[0,9,19,29],[[1,69,72,81],[20,72,74,76],[37,72]],[[1,6,10]])
        llim=mm*64
        ulim=llim+len(pNoteDur)
        #print(mm,pNoteDur,'\n')
        notes_save = GetMelody(Scale,len(pNoteDur),numbers[llim:ulim],InRests,ChLock,MelLock)
        #MusDist = FFTdist(notes_save,pNoteDur,UnivFreq,False)
        #AbsFFT = numpy.abs(MusDist-1.3)
        AbsFFT = MaxJump(notes_save,pNoteDur)
        if AbsFFT<Mx_save:
            pNoteSeq = notes_save
            Mx_save = AbsFFT
        
            
    return pNoteDur,pNoteSeq
        
        
def SoloBuild(Sc,numbers):
    BuildM=[]
    sNote = numbers[0]%len(Sc)
    sDir = numbers[0]%2
    Inflections = [numbers[1]%64,numbers[3]%64,numbers[5]%64]
    Breaks = [15,47]
    TopNoteAllowed=1
    used=0
    for a in range (0,64):
        if a in Breaks:
            BuildM.append(-1)
        else:
            BuildM.append(Sc[sNote])
            if a in Inflections:
                sDir = (sDir+1)%2
            if (sDir==1 and sNote==0) or (sNote==(len(Sc)-1) and sDir==0):
                sDir = (sDir+1)%2
            if sDir==0:
                sNote += 1
            if sDir==1:
                sNote += -1
            if (TopNoteAllowed<2 and sNote==(len(Sc)-1)):
                sNote += -3
                sDir = (sDir+1)%2
                used+=1
            if (TopNoteAllowed==2 and sNote==(len(Sc)-1)) or used==1:
                TopNoteAllowed = (TopNoteAllowed+1)%3
                used=0
    return BuildM
    
def SuperSoloBuild(Sc,idx,rnum):
    Breaks = [15,47]
    Mx_save = 999999
    x_dur=[]
    for a in range (0,64):
        x_dur.append(1)
    Save_FFT=[]
    for mm in range (0,idx):
        x_solo=[]
        xf_solo=[]
        llim=mm*64
        ulim=llim+64
        numbers = rnum[llim:ulim]
        
        sNote = numbers[0]%len(Sc)
        sDir = numbers[0]%2
        Inflections = [numbers[1]%64,numbers[3]%64,numbers[5]%64]
        
        TopNoteAllowed=1
        used=0
        for a in range (0,64):
            if a in Breaks:
                x_solo.append(-1)
                xf_solo.append(saveval)
            else:
                saveval = Sc[sNote]
                x_solo.append(Sc[sNote])
                xf_solo.append(saveval)
                if a in Inflections:
                    sDir = (sDir+1)%2
                if (sDir==1 and sNote==0) or (sNote==(len(Sc)-1) and sDir==0):
                    sDir = (sDir+1)%2
                if sDir==0:
                    sNote += 1
                if sDir==1:
                    sNote += -1
                if (TopNoteAllowed<2 and sNote==(len(Sc)-1)):
                    sNote += -3
                    sDir = (sDir+1)%2
                    used+=1
                if (TopNoteAllowed==2 and sNote==(len(Sc)-1)) or used==1:
                    TopNoteAllowed = (TopNoteAllowed+1)%3
                    used=0
                    
        SoloFFT = FFTdist(xf_solo,x_dur,UnivFreq,False)
        #Save_FFT.append(SoloFFT)
        AbsFFT = numpy.abs(SoloFFT-1.5)
        if AbsFFT<Mx_save:
            pNoteSeq = x_solo
            Mx_save = AbsFFT
    #print(numpy.nanmin(Save_FFT),numpy.nanmax(Save_FFT))
    return pNoteSeq
            
def SuperBuild(Sc,idx,rnum):
    Mx_save = -999999
    x_dur=[]
    ChangeSeq = [-2,-1,1,2]
    Breaks = [15,47]
    for a in range (0,64):
        x_dur.append(1)
    for mm in range (0,idx):
        x_solo=[]
        xf_solo=[]
        llim=mm*64
        ulim=llim+64
        numbers = rnum[llim:ulim]
        CurrIdx = 66# = 72 in midi  numbers[0]%len(Sc)
        for a in range (0,64):
            if a in Breaks:
                x_solo.append(-1)
                xf_solo.append(saveval)
            else:
                saveval = Sc[CurrIdx]
                x_solo.append(saveval)
                xf_solo.append(saveval)
                
                CurrIdx += ChangeSeq[int(numbers[a]/64)]
            #x_solo.append(Sc[numbers[a]%len(Sc)])
        #x_dur = numpy.ones(64)
        SoloFFT = FFTdist(xf_solo,x_dur,UnivFreq,False)
        if SoloFFT>Mx_save:
            pNoteSeq = x_solo
            Mx_save = SoloFFT
    return pNoteSeq
        
        

#more trials but also limit jumps strictly and still include the breaks     longer pan flute sustain 30 io 0.   **suspend before chord transition by knowing 1 ahead and doing a sequence of 4 claps
#second starting position for each "riff"???
#scale width proportional to coherence also...!!!!!
#can add notes to the solo possibility like 2 notes at once maybe
#key up and generate new each time
#swingy style in drums and stuff as alternate version
#do solo WHEN hits coherence...only permitted on certain cycles
#limit number of times highest note can be reached
#could also use beta for solo calculation here and do many simulations...may be overkill. actually yeah...do this every time event triggers "solo"
#spir type celtic mus
#still chi sq of the actual z scores means that come out ea song chg


ChordSeqStrings = ['C','Bb','F','F']
ChordInvStrings = [2,2,3,3]
ChordOctStrings = [-1,-1,-2,-2]
#ChordOctStrings = [0,0,-1,-1]
ChordDurStrings = [16,16,16,16]

ChordSeqPiano = ['C','C','C','C','C','Bb','Bb','Bb','Bb','Bb','F','F','F','F','F','F','F','F','F','F']
ChordInvPiano = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
ChordOctPiano = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
ChordDurPiano = [3,3,4,3,3,3,3,4,3,3,3,3,4,3,3,3,3,4,3,3]

#ChordSeqStrings = ChordSeqStrings*2
#ChordInvStrings = ChordInvStrings*2
#ChordOctStrings = ChordOctStrings*2
#ChordDurStrings = ChordDurStrings*2
#ChordSeqPiano = ChordSeqPiano*2
#ChordInvPiano = ChordInvPiano*2
#ChordOctPiano = ChordOctPiano*2
#ChordDurPiano = ChordDurPiano*2


#Mtrials,(scale),(pxcloud),progression_type,Numbers!!!  -> noteDur noteSeq

"""
ScaleMods = [0,2,3,4,7,9]
ScaleFull = []
for a in range (-5,15):
    ll=a*12
    for b in range (0,len(ScaleMods)):
        ScaleFull.append(ll+ScaleMods[b])
"""

ScaleMods = [60,62,64,67,69,72,74,75,76,79,81,84,81,79,76,75,74,72,69,67,64,62]
ScaleFull = ScaleMods*20

stoRC,stoB = RNG_Bulk()
NoteDur, NoteSeq = GetNotes(1,RNGfreq,stoB)#initializing with trails=1000

#Solo = SoloBuild(SoloScale,stoB)
Solo = SuperSoloBuild(ScaleFull,RNGfreq,stoB)




fs.program_select(0, sfid0, 0, 99)
#fs.cc(0, 1, 0)
#fs.cc(0, 7, 127)
fs.cc(0, 91, 127)
fs.cc(0, 64, 0)

fs1.program_select(0, sfid1, 0, 75)
#fs1.cc(0, 1, 0)
#fs1.cc(0, 7, 127)
fs1.cc(0, 91, 127)
fs1.cc(0, 64, 60)

fs2.program_select(0, sfid2, 0, 50)#50 89 
#fs1.cc(0, 1, 0)
#fs1.cc(0, 7, 127)
fs2.cc(0, 91, 127)
fs2.cc(0, 64, 0)

fs3.program_select(0, sfid3, 24, 0)
#fs1.cc(0, 1, 0)
#fs1.cc(0, 7, 127)
fs3.cc(0, 91, 127)
fs3.cc(0, 64, 0)

fs4.program_select(0, sfid4, 0, 0)
#fs1.cc(0, 1, 0)
#fs1.cc(0, 7, 127)
fs4.cc(0, 91, 127)
fs4.cc(0, 64, 100)


#https://bspaans.github.io/python-mingus/doc/wiki/refMingusMidiPyfluidsynth.html

LoopInf=True
#while LoopInf==True:
ChordOn=False
ChordOnP=False
ModCount=0
cyc=0
Mcyc = None
FutureType = 2

MeanDev = []
SoloCyc = [-1]

#concurrent.futures.ProcessPoolExecutor().submit(RNG_Bulk, 0)
#time.sleep(30)


concurrent.futures.ThreadPoolExecutor().submit(PlayHat)
concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
concurrent.futures.ThreadPoolExecutor().submit(PlayClap)
time.sleep(3)



    





while LoopInf==True:
    cntr=0
    scnt=0
    Mcnt=0
    cntrC=0
    scntC=0
    McntC=0
    cntrCp=0
    scntCp=0
    McntCp=0
    p1 = concurrent.futures.ThreadPoolExecutor().submit(RNG_Bulk)
    for a in range (0,numpy.sum(NoteDur)):
        if cyc in SoloCyc:
            if a%16 in SoloAccentIdx:
                ivolume = 127
            else:
                ivolume = 100
            sid = Solo[a]+ModCount
            fs4.noteon(0, sid, ivolume)
        if (cntr==Mcnt):
            if NoteSeq[scnt]>=0:
                if 0<=cyc<=3:
                    fs4.noteon(0, NoteSeq[scnt],100)
                    fs.noteon(0, NoteSeq[scnt], 80)
                if (cyc>=4) and (cyc not in SoloCyc):
                    fs.noteon(0, NoteSeq[scnt], 50)
                    fs1.noteon(0, NoteSeq[scnt], 127)
            Mcnt=NoteDur[scnt]
            scnt+=1
            cntr=0
        cntr+=1
        
        if cyc>3:
            if (a%4==2):
                concurrent.futures.ThreadPoolExecutor().submit(PlayHat)
            if (a%4==0):
                concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
        if cyc>1:
            if (a%8==4):
                concurrent.futures.ThreadPoolExecutor().submit(PlayClap)
        
    
        if (cntrC==McntC):
            if (ChordOn==True) and (ChordPluck==False):
                fs2.noteoff(0, AN[0])
                fs2.noteoff(0, AN[1])
                fs2.noteoff(0, AN[2])
                #print(AN,ModCount)

            AN = ChordID(ChordSeqStrings[scntC],ChordOctStrings[scntC],ChordInvStrings[scntC])
            AN = numpy.array(AN)+ModCount
            fs2.noteon(0, AN[0], 50)
            fs2.noteon(0, AN[1], 50)
            fs2.noteon(0, AN[2], 50)
            ChordOn=True

            McntC=ChordDurStrings[scntC]
            scntC+=1
            cntrC=0
            
        cntrC+=1
        
        if (cntrCp==McntCp):
            if (ChordOnP==True) and (ChordPluck==False):
                fs3.noteoff(0, ANpiano[0])
                fs3.noteoff(0, ANpiano[1])
                fs3.noteoff(0, ANpiano[2])
                #print(ANpiano)

            ANpiano = ChordID(ChordSeqPiano[scntCp],ChordOctPiano[scntCp],ChordInvPiano[scntCp])
            ANpiano = numpy.array(ANpiano)+ModCount
            if numpy.amax(ANpiano)>=82:
                ANpiano = numpy.array(ANpiano)-12

            if numpy.amax(ANpiano)>=82:
                print(numpy.amax(ANpiano))

            if (cyc>1) and (cyc not in SoloCyc):
                fs3.noteon(0, ANpiano[0], 55)
                fs3.noteon(0, ANpiano[1], 55)
                fs3.noteon(0, ANpiano[2], 55)
            ChordOnP=True

            McntCp=ChordDurPiano[scntCp]
            scntCp+=1
            cntrCp=0
            
        cntrCp+=1
        
        

        time.sleep(PluckSpd)
        fs.noteoff(0, NoteSeq[scnt-1])
        fs1.noteoff(0, NoteSeq[scnt-1])
        fs4.noteoff(0, NoteSeq[scnt-1])
        #fs1.noteoff(0, NoteSeq[scnt-1]+12)
        
        if (ChordOn==True) and (ChordPluck==True):
            fs2.noteoff(0, AN[0])
            fs2.noteoff(0, AN[1])
            fs2.noteoff(0, AN[2])
            fs3.noteoff(0, ANpiano[0])
            fs3.noteoff(0, ANpiano[1])
            fs3.noteoff(0, ANpiano[2])
        
        #time.sleep((NoteDur[a]*NoteSpeed)+((NoteDur[a]-1)*PluckSpd))
        time.sleep(NoteSpeed)
    
    #time.sleep(0.2)
    
    stoRC,stoB = p1.result()
    
    MeanDev.append(stoRC)
    
    ActionZ = (stoRC-(256*RNGfreq))/((512*RNGfreq*0.25)**0.5)
    
    TotalDev = numpy.sum(MeanDev)
    NDev = len(MeanDev)
    MeanZ = (TotalDev-(NDev*256*RNGfreq)) / ((NDev*512*RNGfreq*0.25)**0.5)
    if (numpy.abs(MeanZ)>=SoloZ) and ((cyc-SoloCyc[-1])>8):
        SoloCyc = [cyc+1,cyc+2,cyc+3,cyc+4]
    
    cyc+=1
    if ActionZ >= ThresholdZ:
        
        if cyc not in SoloCyc:
            
            cntr=0
            scnt=0
            Mcnt=0
            cntrC=0
            scntC=0
            McntC=0
            cntrCp=0
            scntCp=0
            McntCp=0
            for a in range (0,numpy.sum(NoteDur)):
                #if cyc in SoloCyc:
                #    sid = Solo[a]+ModCount
                #    fs4.noteon(0, sid, 100)
                if (cntr==Mcnt):
                    if NoteSeq[scnt]>=0:
                        if 0<=cyc<=3:
                            fs4.noteon(0, NoteSeq[scnt],100)
                            fs.noteon(0, NoteSeq[scnt], 80)
                        if (cyc>=4) and (cyc not in SoloCyc):
                            fs.noteon(0, NoteSeq[scnt], 50)
                            fs1.noteon(0, NoteSeq[scnt], 127)
                    Mcnt=NoteDur[scnt]
                    scnt+=1
                    cntr=0
                cntr+=1
                
                if a<32:
                    if (a%4==2):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayHat)
                    if (a%4==0):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
                    if (a%8==4):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayClap)
                else:
                    if (a%4==0):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
                
            
                if (cntrC==McntC) and (a<32):
                    fs2.noteoff(0, AN[0])
                    fs2.noteoff(0, AN[1])
                    fs2.noteoff(0, AN[2])
        
                    AN = ChordID(ChordSeqStrings[scntC],ChordOctStrings[scntC],ChordInvStrings[scntC])
                    AN = numpy.array(AN)+ModCount
                    fs2.noteon(0, AN[0], 50)
                    fs2.noteon(0, AN[1], 50)
                    fs2.noteon(0, AN[2], 50)
        
                    McntC=ChordDurStrings[scntC]
                    scntC+=1
                    cntrC=0
                    
                cntrC+=1
                
                if (cntrCp==McntCp):
                    fs3.noteoff(0, ANpiano[0])
                    fs3.noteoff(0, ANpiano[1])
                    fs3.noteoff(0, ANpiano[2])
        
                    ANpiano = ChordID(ChordSeqPiano[scntCp],ChordOctPiano[scntCp],ChordInvPiano[scntCp])
                    ANpiano = numpy.array(ANpiano)+ModCount
                    if numpy.amax(ANpiano)>=82:
                        ANpiano = numpy.array(ANpiano)-12
        
                    if numpy.amax(ANpiano)>=82:
                        print(numpy.amax(ANpiano))
        
                    if (cyc>1) and (cyc not in SoloCyc):
                        fs3.noteon(0, ANpiano[0], 55)
                        fs3.noteon(0, ANpiano[1], 55)
                        fs3.noteon(0, ANpiano[2], 55)
                    ChordOnP=True
        
                    McntCp=ChordDurPiano[scntCp]
                    scntCp+=1
                    cntrCp=0
                    
                cntrCp+=1
                
                
        
                time.sleep(PluckSpd)
                fs.noteoff(0, NoteSeq[scnt-1])
                fs1.noteoff(0, NoteSeq[scnt-1])
                fs4.noteoff(0, NoteSeq[scnt-1])
                #fs1.noteoff(0, NoteSeq[scnt-1]+12)
                
                if (ChordOn==True) and (ChordPluck==True):
                    fs2.noteoff(0, AN[0])
                    fs2.noteoff(0, AN[1])
                    fs2.noteoff(0, AN[2])
                    fs3.noteoff(0, ANpiano[0])
                    fs3.noteoff(0, ANpiano[1])
                    fs3.noteoff(0, ANpiano[2])
                
                #time.sleep((NoteDur[a]*NoteSpeed)+((NoteDur[a]-1)*PluckSpd))
                time.sleep(NoteSpeed)
            
            
        
        
        
        if numpy.amax(NoteSeq)>=85:
            NoteSeq = numpy.array(NoteSeq)+(ModAmount-12)
        else:
            NoteSeq = numpy.array(NoteSeq)+ModAmount
        ModCount += ModAmount


        fs2.noteoff(0, AN[0])
        fs2.noteoff(0, AN[1])
        fs2.noteoff(0, AN[2])
        fs3.noteoff(0, ANpiano[0])
        fs3.noteoff(0, ANpiano[1])
        fs3.noteoff(0, ANpiano[2])
        outfile.write("expect up at %d\n"%(time.time()))
        
    if ActionZ <= (-1*ThresholdZ):
        
        
        
        if cyc>6:
            
            cntr=0
            scnt=0
            Mcnt=0
            cntrC=0
            scntC=0
            McntC=0
            cntrCp=0
            scntCp=0
            McntCp=0
            for a in range (0,numpy.sum(NoteDur)):
                #if cyc in SoloCyc:
                #    sid = Solo[a]+ModCount
                #    fs4.noteon(0, sid, 100)
                if (cntr==Mcnt):
                    if NoteSeq[scnt]>=0:
                        fs4.noteon(0, NoteSeq[scnt],100)
                    Mcnt=NoteDur[scnt]
                    scnt+=1
                    cntr=0
                cntr+=1
                
                if a<32:
                    if (a%4==2):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayHat)
                    if (a%4==0):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
                    if (a%8==4):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayClap)
                else:
                    if (a%4==0):
                        concurrent.futures.ThreadPoolExecutor().submit(PlayKick)
                
            
                if (cntrC==McntC) and (a<32):
                    fs2.noteoff(0, AN[0])
                    fs2.noteoff(0, AN[1])
                    fs2.noteoff(0, AN[2])
        
                    AN = ChordID(ChordSeqStrings[scntC],ChordOctStrings[scntC],ChordInvStrings[scntC])
                    AN = numpy.array(AN)+ModCount
                    fs2.noteon(0, AN[0], 50)
                    fs2.noteon(0, AN[1], 50)
                    fs2.noteon(0, AN[2], 50)
        
                    McntC=ChordDurStrings[scntC]
                    scntC+=1
                    cntrC=0
                    
                cntrC+=1
                

                    
                cntrCp+=1
                
                
        
                time.sleep(PluckSpd)
                fs.noteoff(0, NoteSeq[scnt-1])
                fs1.noteoff(0, NoteSeq[scnt-1])
                fs4.noteoff(0, NoteSeq[scnt-1])
                #fs1.noteoff(0, NoteSeq[scnt-1]+12)
                
                if (ChordOn==True) and (ChordPluck==True):
                    fs2.noteoff(0, AN[0])
                    fs2.noteoff(0, AN[1])
                    fs2.noteoff(0, AN[2])
                    fs3.noteoff(0, ANpiano[0])
                    fs3.noteoff(0, ANpiano[1])
                    fs3.noteoff(0, ANpiano[2])
                
                #time.sleep((NoteDur[a]*NoteSpeed)+((NoteDur[a]-1)*PluckSpd))
                time.sleep(NoteSpeed)
        
        
        
        ModCount = 0
        ScaleZ = int((numpy.abs(MeanZ)/2.0)*RNGfreq)
        #print(MeanZ)
        NOW = time.time()
        print(NOW)
        if ScaleZ==0:
            ScaleZ=1
        if ScaleZ > RNGfreq:
            ScaleZ = RNGfreq
        MeanDev = []
        
        SoloCyc = [-1]
        
        fs2.noteoff(0, AN[0])
        fs2.noteoff(0, AN[1])
        fs2.noteoff(0, AN[2])
        fs3.noteoff(0, ANpiano[0])
        fs3.noteoff(0, ANpiano[1])
        fs3.noteoff(0, ANpiano[2])
        #p2 = concurrent.futures.ThreadPoolExecutor().submit(GetNotes,1,1000,stoB)
        #Mcyc = cyc+1
        
        #if cyc==Mcyc:
        #Mcyc = None
        #NoteDur,NoteSeq = p2.result()
        
        NextType = FutureType
        NoteDur,NoteSeq = GetNotes(NextType,ScaleZ,stoB)
        Solo = SuperSoloBuild(ScaleFull,ScaleZ,stoB)
        ult_dist,ult_rep = AbsDist(NoteSeq)#can remove this line later when don't need to write it below ... consider changing minimum rep
        ult_fft = FFTdist(NoteSeq,NoteDur,UnivFreq,False)
        ult_jump = MaxJump(NoteSeq,NoteDur)
        
        outfile.write("changetune,%d,%d,%f,%f,%f,%f,%f,%s\n"%(NOW,ScaleZ,MeanZ,ult_dist,ult_rep,ult_fft,ult_jump,NoteDur))
        for b in range (0,len(stoB)):
            outfile.write('%d,'%(stoB[b]))
        
        if NextType==1:

            ChordSeqStrings = ['C','Bb','F','F']
            ChordInvStrings = [2,2,3,3]
            ChordOctStrings = [-1,-1,-2,-2]
            ChordDurStrings = [16,16,16,16]
            
            ChordSeqPiano = ['C','C','C','C','C','Bb','Bb','Bb','Bb','Bb','F','F','F','F','F','F','F','F','F','F']
            ChordInvPiano = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
            ChordOctPiano = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
            ChordDurPiano = [3,3,4,3,3,3,3,4,3,3,3,3,4,3,3,3,3,4,3,3]
            
            FutureType = 2
        
        if NextType==2:
            ChordSeqStrings = ['Am','F','C','G']#consider inversions
            ChordInvStrings = [1,2,3,1]
            ChordOctStrings = [-1,-2,-2,-2]
            ChordDurStrings = [16,16,16,16]
            
            ChordSeqPiano = ['Am','Am','Am','Am','Am','F','F','F','F','F','C','C','C','C','C','G','G','G','G','G']
            ChordInvPiano = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,1,1,1,1]
            ChordOctPiano = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ChordDurPiano = [3,3,4,3,3,3,3,4,3,3,3,3,4,3,3,3,3,4,3,3]
            
            FutureType = 1
            
        cyc=0
    
    
    #if _==5:
    #    p2 = concurrent.futures.ThreadPoolExecutor().submit(GetNotes,1,1000,stoB)
    #if _==8:
    #    NoteDur,NoteSeq = p2.result()
        
if (ChordOn==True):
    fs2.noteoff(0, AN[0])
    fs2.noteoff(0, AN[1])
    fs2.noteoff(0, AN[2])
    fs3.noteoff(0, ANpiano[0])
    fs3.noteoff(0, ANpiano[1])
    fs3.noteoff(0, ANpiano[2])

outfile.close()

#claps > snap 3
#kicks > kick 5

#fs.delete()

