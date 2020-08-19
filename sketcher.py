#!/usr/bin/env python
#
# sketcher.py 
#

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import sys
import random

black = (0,0,0)
white = (255,255,255)

def map_value_to_level(value) :
    if (value < 26) :
        return 0
    elif (value < 52) :
        return 1
    elif (value < 79) :
        return 2
    elif (value < 110) :
        return 3
    elif (value < 135) :
        return 4
    elif (value < 160) :
        return 5
    elif (value < 185) :
        return 6
    elif (value < 210) :
        return 7
    else :
        return 8

def load_hatches(num_shades) :
    hlist = []
    for i in range(num_shades) :
        #fname = "images/hatch" + str(i) + ".png"
        fname = "images/hatch" + str(i) + ".bmp"
        img = Image.open(fname)
        hlist.append( img.load() )
    return hlist

def maybe_black_dot(pixel, i, j, greyval, threshold, dot_size) :
    if (greyval > threshold) :
        pixel[i,j] = black
        if (dot_size == '2' or dot_size == '3' or dot_size == '4') :
            if (j > 0) :
            	pixel[i,j-1] = black
        if (dot_size == '3' or dot_size == '4') :
            if (j > 0 and i > 0) :
                pixel[i-1,j-1] = black
        if (dot_size == '4') :
            if (i > 0) :
                pixel[i-1,j] = black
    else :
    	pixel[i,j] = white
	
if __name__ == '__main__':

    try :
        filename = sys.argv[1]
        print( 'trying to open ' + filename )
    except :
        print( 'failed to open image file specified in 1st argument!' )
        filename = 'images/seven11.png'
        print( 'will try to open ' + filename + ' instead' )

    img = Image.open(filename)
    
    try :
        paraminfo = sys.argv[3]
    except : 
        paraminfo = ""

    try :
        filter_name = sys.argv[2]
    except :
        print( 'no filter spec found in 2nd argument, using SK' )
        filter_name = "SK"
           
    apply = False # default to no filter

    if filter_name == "FE" :
        filter = ImageFilter.FIND_EDGES
        apply = True
    elif filter_name == "EE" :
        filter = ImageFilter.EDGE_ENHANCE
        apply = True
    elif filter_name == "EEM" :
        filter = ImageFilter.EDGE_ENHANCE_MORE
        apply = True
    elif filter_name == "EM" :
        filter = ImageFilter.EMBOSS
        apply = True
    elif filter_name == "SM" :
        filter = ImageFilter.SMOOTH
        apply = True
    elif filter_name == "SMM" :
        filter = ImageFilter.SMOOTH_MORE
        apply = True
    elif filter_name == "BL" :
        filter = ImageFilter.BLUR
        apply = True
    elif filter_name == "SH" :
        filter = ImageFilter.SHARPEN
        apply = True
    elif filter_name == "CO" : 
        filter = ImageFilter.CONTOUR
        apply = True
    elif filter_name == "DE" : 
        filter = ImageFilter.DETAIL
        apply = True
    elif filter_name == "MX" : 
        filter = ImageFilter.MaxFilter
        apply = True
    elif filter_name == "MN" : 
        filter = ImageFilter.MinFilter
        apply = True
    elif filter_name == "ME" : 
        filter = ImageFilter.MedianFilter
        apply = True
    elif filter_name == "MO" : 
        filter = ImageFilter.ModeFilter
        apply = True
    elif filter_name in ("EF", "EF2", "EF3", "EF4") :
        filter = ImageFilter.FIND_EDGES  # but produce black on white instead of white on black    
        apply = True
    elif filter_name == "ST" : # stipple
        pass
    elif filter_name == "ES" : # enhanced stipple
        pass
    elif filter_name == "LT" : # lighten
        pass
    elif filter_name == "WH" : # whiten
        pass
    elif filter_name == "CH" : # crosshatch (TBD)
        pass
    elif filter_name in ("AC", "EQ", "EQM", "FC", "FS", "GC", "GS", "HI", "IV", "PO", "QG", "QM", "QT", "SK") :
        pass
    else :
        # default to earn this code the name "sketcher"
        print( 'failed to recognize filter name in 2nd arg (' + filter_name + '), using SK' )
        filter_name = "SK"
    
    print( 'filter: ' + filter_name )
    #print( 'debug: ' + filter_name[0:2])

    if (filter_name == "HI") :  # histogram
        print(img.histogram())
        exit(0)

    if (filter_name == "GC") :  # get colors
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 200
        paraminfo = str(threshold)
        print(img.getcolors(threshold))
        exit(0)

    if apply :
        print( 'applying filter' )
        img = img.filter(filter)

    if (filter_name == "AC") :  # autocontrast
        try :
            cutoff = int(paraminfo)
            print( cutoff )
        except :
            cutoff = 10
        paraminfo = '-' + str(cutoff)
        img = ImageOps.autocontrast(img, cutoff)
    
    elif (filter_name[0:2] == "EF") :  # edge find
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 200
        paraminfo = str(threshold)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        try :
            (r,g,b) = pixel[0,0]
            monochrome = False
        except :
            monochrome = True
        if (monochrome) :
            for i in range(width) :
                for j in range(height) :
                    greyval = pixel[i,j]
                    # TO DO
        else :
            for i in range(width) :
                for j in range(height) :
                    (r,g,b) = pixel[i,j]
                    greyval = (r+g+b)/3
                    dot_size = '1' if len(filter_name) < 3 else filter_name[2]
                    maybe_black_dot(pixel, i, j, greyval, threshold, dot_size)

    elif (filter_name == "EQM") :  # equalize monochrome?
        mono_incr = 12.8
        rgb_incr = (3*255)/10.0
        grey = [0]
        brightness = [0]
        for lev in range(1,10) :
            grey.append(lev*mono_incr)
            brightness.append(lev*rgb_incr)
            print(grey[-1], brightness[-1])
    
    elif (filter_name == "ES") :  # enhanced stipple (uses 2 thresholds)
        try :
            leave_black = int(paraminfo)
            print( leave_black )
        except :
            leave_black = 200
        try :
            leave_white = int(sys.argv[4])
            print( leave_white )
        except :
            leave_white = 600
        paraminfo = str(leave_black) + '-' + str(leave_white)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                (r,g,b) = pixel[i,j]
                brightness = r+g+b
                if (brightness > leave_white) :
                    pixel[i,j] = white
                elif (brightness < leave_black) :
                    pixel[i,j] = black
                else :
                    test = random.randrange(leave_black, leave_white)
                    pixel[i,j] = black if (brightness < test) else white

    elif (filter_name == "EQ") :  # equalize
        img = ImageOps.equalize(img)
    
    elif (filter_name == "FC") :  # false color
        try :
            blk_threshold = int(paraminfo)
            print( blk_threshold )
        except :
            blk_threshold = 20
        try :
            wht_threshold = int(sys.argv[4])
            print( wht_threshold )
        except :
            wht_threshold = 200
        paraminfo = str(blk_threshold) + '-' + str(wht_threshold)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                try :
                    (r,g,b) = pixel[i,j]
                    value = r+g+b
                    if value < blk_threshold :
                        pixel[i,j] = black
                    elif value > wht_threshold :
                        pixel[i,j] = white
                    else :
                        if (r >= g) and (r >= b) :
                            pixel[i,j] = (255,0,0)
                        elif (g >= r) and (g >= b) :
                            pixel[i,j] = (0,255,0)
                        else :
                            pixel[i,j] = (0,0,255)
                except :
                    greyval = pixel[i,j]
                    pixel[i,j] = 0 if (greyval < blk_threshold) else (255 if (greyval > wht_threshold) else 128)
        
    elif (filter_name == "FS") :  # fat stipple (4-pixel squares)
        try :
            blk_threshold = int(paraminfo)
            print( blk_threshold )
        except :
            blk_threshold = 60
        try :
            wht_threshold = int(sys.argv[4])
            print( wht_threshold )
        except :
            wht_threshold = 600
        try :
            blackness = int(sys.argv[5])
            print( blackness )
        except :
            blackness = 300
        paraminfo = str(blk_threshold) + '-' + str(wht_threshold) + '-' + str(blackness)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                try :
                    (r,g,b) = pixel[i,j]
                    value = r+g+b
                    if value < blk_threshold :
                        pixel[i,j] = black
                    elif value > wht_threshold :
                        pixel[i,j] = white
                    else :
                        if (value < random.randrange(0,blackness)):
                            pixel[i,j] = black
                            pixel[i,j-1] = black
                            pixel[i-1,j-1] = black
                            pixel[i-1,j] = black
                        else :
                            pixel[i,j] = white
                except :
                    greyval = pixel[i,j]
                    pixel[i,j] = 0 if (greyval < blk_threshold) else (255 if (greyval > wht_threshold) else 128)
    
    elif (filter_name == "GS") :  # greyscale
        img = ImageOps.grayscale()
    
    elif (filter_name == "IV") :  # invert
        img = ImageOps.invert(img)
    
    elif (filter_name == "LT") :  # lighten
        try :
            pct_increase = int(paraminfo)
            print( pct_increase )
        except :
            pct_increase = 10
        paraminfo = str(pct_increase)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                (r,g,b) = pixel[i,j]
                r = min(int(r*(100+pct_increase)/100.), 255)
                g = min(int(g*(100+pct_increase)/100.), 255)
                b = min(int(b*(100+pct_increase)/100.), 255)
                pixel[i,j] = (r,g,b) 

    elif (filter_name == "QG") :  # quantize greyscale (quantize, then grayscale)
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 20
        img = ImageOps.grayscale(img)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
        img = img.convert("RGB")
        
    elif (filter_name == "QM") :  # quantize monochrome (grayscale, then quantize)
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 20
        img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
        img = ImageOps.grayscale(img)
        img = img.convert("RGB")
        
    elif (filter_name == "QT") :  # quantize
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 20
        paraminfo = str(threshold)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
        img = img.convert("RGB")
                    
    elif (filter_name == "PO") :  # posterize
        try :
            nbits = int(paraminfo)
            print( nbits )
        except :
            nbits = 4
        paraminfo = '-' + str(nbits)
        img = ImageOps.posterize(img, nbits)
    
    elif (filter_name == "SK") :  # sketch
        try :
            num_shades = int(paraminfo)
            print( num_shades )
        except :
            num_shades = 9
        try :
            offset = int(sys.argv[4])
            print( offset )
        except :
            offset = 0
        paraminfo = str(num_shades) + '-' + str(offset)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        hatch = load_hatches(num_shades)
        for i in range(width) :
            for j in range(height) :
                try :
                    (r,g,b) = pixel[i,j]
                    value = (r+g+b)/3 + offset
                    h = map_value_to_level(value)
                    pixel[i,j] = hatch[h][i % 20, j % 20]
                    # pixel[i,j] = hatch[h][random.randrange(0,20),random.randrange(0,20)] 
                except :
                    greyval = pixel[i,j]
                    pixel[i,j] = 0 if (greyval > threshold) else 255
        
    elif (filter_name == "ST") :  # stipple
        try :
            threshold = int(paraminfo)
            print( threshold )
        except :
            threshold = 200
        paraminfo = str(threshold)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                (r,g,b) = pixel[i,j]
                test = random.randrange(0,threshold)
                pixel[i,j] = black if (r+g+b < test) else white 

    elif (filter_name == "WH") :  # whiten
        try :
            addend = int(paraminfo)
            print( addend )
        except :
            addend = 20
        paraminfo = str(addend)
        pixel = img.load()
        (width, height) = img.size
        print( width, height )
        for i in range(width) :
            for j in range(height) :
                (r,g,b) = pixel[i,j]
                r = min(r+addend, 255)
                g = min(g+addend, 255)
                b = min(b+addend, 255)
                pixel[i,j] = (r,g,b) 

    try :
        left_part = sys.argv[1].split('/')[-1]
        print("1")
        left_part = left_part[:-4]
        print("2")
        newfile = 'newimages/' + left_part + '-' + filter_name + paraminfo + '.jpg'
        print( 'writing to ' + newfile )
        img.save(newfile)
        print("3")
    except :
        print( 'writing to default.jpg' )
        img.save('default.jpg')
    
