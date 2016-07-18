#code to read from data  using Arduino
import serial, io, time
import sys, select
import logging, os


#print "Testing serial interface with ENS2020"
#content = dir(serial)
#print content

folder_path = os.getcwd()
Subject_name ='TT'
Session_num = 1
Block_num = 0 #change

#myfilename1 = os.getcwd() + '/NJBT_FD_impulse_input_block5_' + time.strftime("%m-%d-%Y_%H_%M_%S.txt")
#myfilename2 = os.getcwd() + '/NJBT_FD_impulse_response_block5_' + time.strftime("%m-%d-%Y_%H_%M_%S.txt")
myfilename1 = folder_path + '/' + Subject_name + '_session' + str(Session_num) + '_block' + str(Block_num) + '_' + time.strftime("%m-%d-%Y_%H_%M_%S.txt")

f_force_response_obj = open(myfilename1,'w')
f_force_response_obj.write("""Force response, sampling frequency 100 Hz.
File created on: """)
f_force_response_obj.write(time.strftime("%m-%d-%Y_%H:%M:%S")+'\r\n')

# ls -al /dev/tty*  ---- To list usb-com port 
ser = serial.Serial('/dev/ttyACM1',115200, timeout = 1) # Arduino
#ser = serial.Serial('/dev/ttyUSB0',19200, timeout = 1) # ENS2020
    
if ser.inWaiting() != 0:
    ser.flushInput()
    
command = ""    
while True:
    while ser.inWaiting() != 0:
        output = ser.readline()
        if len(output) >= 10:
            f_force_response_obj.write(output)
            print output
        else:
            print output
        #logging.info(output)
        #print '.',
        #f_obj.write(output)
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
f_force_response_obj.close()
print "End of program."
