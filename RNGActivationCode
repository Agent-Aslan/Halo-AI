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

#testfile = open('testLIFXRNG_%d.txt'%(int(time.time()*1000)),'w')
#testfile.close()

#sleep(30)

ports=dict()  
ports_avaiable = list(list_ports.comports())
rng_com_port = None
for temp in ports_avaiable:
    if temp[1].startswith("TrueRNG"):
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

starttime = int(time.time()*1000)

outfile = open('/home/bridge/RNG_%d.txt'%(starttime),'w')



def main():

    print("Breathing...")
    while True:
        ser.flushInput()
        x = ser.read(1000)
        #print(x)
        #hues = []
        #for a in range (0,1000):
        #    hues.append(x[a])
        #rng_zones = [(hues[0], 65535, 20000, 3500),(hues[1], 65535, 20000, 3500),(hues[2], 65535, 20000, 3500),(hues[3], 65535, 20000, 3500),(hues[4], 65535, 20000, 3500),(hues[5], 65535, 20000, 3500),(hues[6], 65535, 20000, 3500),(hues[7], 65535, 20000, 3500),(hues[0], 65535, 20000, 3500),(hues[1], 65535, 20000, 3500),(hues[2], 65535, 20000, 3500),(hues[3], 65535, 20000, 3500),(hues[4], 65535, 20000, 3500),(hues[5], 65535, 20000, 3500),(hues[6], 65535, 20000, 3500),(hues[7], 65535, 20000, 3500)]
        #strip.set_zone_colors(rng_zones, 2000, True)
        #outfile = open('LIFXRNG_%d.txt'%(starttime),'a')
        #if (countcycle<100):
        for a in range (0,len(x)):
            outfile.write('%d,'%(x[a]))
        outfile.write('%d\n'%(int(time.time()*1000)))
        #if (countcycle==100):
        #    outfile.close()
        #countcycle+=1
        #outfile.flush()
        #os.fsync(outfile.fileno())
        sleep(1)

            
"""
    bulb = devices[0]
    print("Selected {}".format(bulb.get_label()))

    # get original state
    print("Turning on all lights...")
    original_power = bulb.get_power()
    original_color = bulb.get_color()
    bulb.set_power("on")

    sleep(1) # for looks

    print("Flashy fast rainbow")
    rainbow(bulb, 0.1)

    print("Smooth slow rainbow")
    rainbow(bulb, 1, smooth=True)

    print("Restoring original power and color...")
    # restore original power
    bulb.set_power(original_power)
    # restore original color
    sleep(0.5) # for looks
    bulb.set_color(original_color)
"""



if __name__=="__main__":
    main()
