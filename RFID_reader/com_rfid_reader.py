# -*- coding: utf-8 -*-
import time
import serial
from time import sleep
SCREEN_DISPLAY=True
DELIMITER=','

#SERIAL_PORT='/dev/ttyACM0' # serial port terminal
SERIAL_PORT='COM3'

file_name= 'RFID_result.txt'
fid= open(file_name,'w')

scale=serial.Serial(SERIAL_PORT,timeout=10,baudrate=9600)
batch_box=0
box_rfid = []
print('RFID를 태그해주세요')
while batch_box<=3:
    str_scale=scale.readline()
    time_now=time.strftime("%Y-%m-%d %H:%M:%S")
    if SCREEN_DISPLAY:
        # print(str.encode(time_now+DELIMITER)+str_scale)
        text = str.encode(time_now+DELIMITER)+str_scale
        text = text.decode('utf-8')
        # print(type(text))
        if 'Card UID' in text:
            box_rfid.append(text)
            batch_box+=1
            print('{}개 인식 완료'.format(batch_box))

scale.close()


for i in box_rfid:
    i = i.replace('\n', '')
    i = i.split(',')[-1]
    i = i.split(':')[-1]
    fid.write(i)

fid.close()
print('종료')