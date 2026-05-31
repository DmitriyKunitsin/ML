import gc
import asyncio
import time
from machine import Pin
from onewire import OneWire
from ds18x20 import DS18X20


class TemperatureSensors:

    def __init__(self, onewire: OneWire, addresses_hex: list[str]):
        self.temp = DS18X20(onewire)
        roms = self.temp.scan()
        self.addresses_hex = addresses_hex
        assert len(roms) == len(addresses_hex), \
            "Len of temperature addresses is not equal of entered addressess list"
        for rom in roms:
            assert rom.hex() in addresses_hex, f"Not found temperature sensor: {rom.hex()}"

    async def aread_temperature(self) -> dict[str, float]:
        try:
            self.temp.convert_temp()
        except Exception:
            raise Exception("Didn't convert with temperature sensors")
        d = dict()
        await asyncio.sleep(0.80)
        for addr in self.addresses_hex:
            try:
                d[addr] = self.temp.read_temp(bytearray.fromhex(addr))
            except Exception:
                raise Exception(f"Didn't read from temperature sensor: {addr}")
        return d

    def read_temperature(self) -> dict[str, float]:
        try:
            self.temp.convert_temp()
        except Exception:
            raise Exception("Didn't convert with temperature sensors")
        d = dict()
        time.sleep(0.75)
        for addr in self.addresses_hex:
            try:
                d[addr] = self.temp.read_temp(bytearray.fromhex(addr))
            except Exception:
                raise Exception(f"Didn't read from temperature sensor: {addr}")
        return d

