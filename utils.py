import struct
import numpy as np
EPS = 0.0000001
ST = 1

def dec_to_2s(number, width):
    if number > 0:
        result = bin(number)
        pad_l = width - (len(result) - 2)
        if pad_l < 0:
            return "Error not enough bits"
        else:
            return "0" * pad_l + result[2:]
    elif number == 0:
        return "0" * width
    else:
        return dec_to_2s(number + 2 ** width, width)

def float_to_hex(number):
    s = struct.pack('>f',number)
    bits = struct.unpack('>l',s)[0]
    return hex(int(dec_to_2s(bits,32),2)).upper().replace("X","f")

def half_to_hex(number):
    s = hex(np.float16(number).view('H'))[2:].zfill(4)
    return "0x" + s + s

