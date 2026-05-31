import ds18x20
import onewire
from dht import DHT22
import time
import os
from sdcard import SDCard
import machine
from machine import Pin, SPI
spi = SPI(1)
sd = SDCard(spi, Pin(15))
os.mount(sd, '/sd')
os.listdir()
os.listdir("/sd")


# the device is on GPIO12
dat = machine.Pin(5)

# create the onewire object
ds = ds18x20.DS18X20(onewire.OneWire(dat))

# scan for devices on the bus
ds_roms = ds.scan()
print('found devices:', ds_roms)


dht = DHT22(Pin(4))
f_sd, f_vfs = "sd/temp.txt", "temp.txt"
try:
    os.remove(f_vfs)
    os.remove(f_sd)
except Exception as e:
    e

def temp():
    # loop 10 times and print all temperatures
    for i in range(10):
        print('temperatures:', end=' ')
        ds.convert_temp()
        dht.measure()
        time.sleep_ms(1000)
        for rom in ds_roms:
            data = list(time.localtime())
            msg = f"ds18:{rom} {data[::-1][3]}:{data[::-1][2]} temp={ds.read_temp(rom):2f}\n " + \
                f"{dht.temperature()=} {dht.humidity()=}"
            print(msg)
            with open(f_sd, "a") as f:
                f.write(msg + "\n")
            # with open(f_vfs, "a") as f:
            #     f.write(msg + "\n")

while True:
    temp()
