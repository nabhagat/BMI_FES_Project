#code to read from data  using Arduino
import serial, io, time
import sys, select
import logging, os


#print "Testing serial interface with ENS2020"
#content = dir(serial)
#print content

folder_path = os.getcwd()

myfilename = os.getcwd() + '/NJBT_ses1_cond2_block3_' + time.strftime("%m-%d-%Y_%H_%M_%S.txt")
f_obj = open(myfilename,'w')

#logging.basicConfig(filename = myfilename + time.strftime("%m-%d-%Y_%H_%M_%S.txt"),loglevel = logging.NOTSET, format = '%(message)s')

# ls -al /dev/tty*  ---- To list usb-com port 
ser = serial.Serial('/dev/ttyACM0',115200, timeout = 1) # Arduino
#ser = serial.Serial('/dev/ttyUSB0',19200, timeout = 1) # ENS2020
    
if ser.inWaiting() != 0:
    ser.flushInput()
    
command = ""    
while True:
    while ser.inWaiting() != 0:
        output = ser.readline()
        print output
        #logging.info(output)
        #print '.',
        f_obj.write(output)
    device_i, device_o, device_e = select.select( [sys.stdin], [], [], 0.1)
    if (device_i):
        command = sys.stdin.readline().strip()
        if command == 'q':
            break
        else:
            #ser.write(command + "\n\r") # For ENS
            #ser.write(command + "\r") # For Controlino 
            ser.write(command + "\n")
            #print "\ncommand sent"       
ser.close()
f_obj.close()
print "End of program."
