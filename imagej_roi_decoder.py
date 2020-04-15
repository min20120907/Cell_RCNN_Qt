import os
import struct
# ImageJ decoder


def int_to_hex(i):  # for 16 bit integers
    return('{:04x}'.format(i).encode())


def float_to_hex(f):  # for 32 bit floats
    return (hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]).encode()


def write_imagej_roi(coordinates, name, path, stroke_width=3, stroke_col='88FFFF00', fill_col='00000000'):

    # Creates ImageJ compatible ROI files for points or circles
    # Parameters:
    #	coordinates:		for circles (t,l,b,r) or (x,y,r) for points (x,y)
    #	name:				Name of the roi (and exported file)
    #	path:				path to write the output files
    # Optional parameters:
    #	stroke_width [int]
    #	stroke_col [0xAARRGGBB]
    #
    # Return: path to output file

    pasfloat = False
    offset_filename = 128
    if len(coordinates) == 4:  # Bounding box
        type = '\x01\x00'  # For rectangles
        # type='\x02\x00'
        top, left, bottom, right = coordinates
    elif len(coordinates) == 3:  # x,y,r : Circle
        type = '\x02\x00'
        x, y, r = coordinates
        top, left, bottom, right = int(
            round(y-r)), int(round(x-r)), int(round(y+r)), int(round(x+r))
    elif len(coordinates) == 2:  # x,y: Point
        type = '\x01\x00'  # This is actualy a one pixel rectangle
        x, y = coordinates
        top, left, bottom, right = int(x), int(y), int(x+1), int(y+1)
        # add subpixel resolution (here it's a real point)
        pasfloat = (not (int(x) == x and int(y) == y))

    filelength = 128+len(name)*2+pasfloat*8
    print(filelength)
    data = bytearray(filelength)

    data[0:4] = '\x49\x6F\x75\x74'.encode()  # "Iout" 0-3
    data[4:6] = '\x00\xE3'.encode()  # Version 4-5

    # roi type   6-7     # Ovals/points
    data[6:8] = type.encode()
    data[8:10] = int_to_hex(top)     				# top      8-9
    data[10:12] = int_to_hex(left)    				# left    10-11
    data[12:14] = int_to_hex(bottom)  				# bottom  12-13
    data[14:16] = int_to_hex(right)   				# right   14-15
    data[34:36] = int_to_hex(stroke_width)  		    # Stroke Width  34-35
    data[40:44] = stroke_col.encode()			# Storke Color 40-43
    data[44:48] = fill_col.encode()
    data[60:64] = '\x00\x00\x00\x40'.encode()					# header2offset 60-63

    if (pasfloat):
        offset_filename = 128+12  # header2offset +12
        data[6: 8] = ' \x0A\x00'
        data[16:18] = '\x00\x01' 						# Marker for 1 exact coordinate
        data[50:52] = '\x00\x80' 						# set options SUB_PIXEL_RESOLUTION
        data[68:72] = float_to_hex(x)
        data[72:72] = float_to_hex(y)
        data[60:64] = '\x00\x00\x00\x4C'.encode()
        data[94:96] = '\x00\x8C'.encode() 						# Name offset
        data[98:100] = int_to_hex(len(name))			# Name Length

    else:
        data[82:84] = '\x00\x80'.encode()						# Name offset
        data[86:88] = int_to_hex(len(name))			# Name Length

    p = offset_filename								# add name
    for c in name:
        print("p and c:", p, c, "\ndata:", data)
        data[p] = 0x00
        data[p+1] = c
    p = p+2
    print(data)
    filename = os.path.join(path, name+".roi")			# write file
    file = open(filename, 'wb')
    file.write(data)
    file.close()
    return(filename)
