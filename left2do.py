#
# left 2 DO from sketcher.py
#

def TEST(filter_name) :
    
    if (filter_name[0:2] == "EF") :  # "edge find" seems like a misnomer here,
                                     # but ImageFilter.FIND_EDGES has been applied
                                     # to the image (in img) before we get here
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

    elif (filter_name == "EQM") :  # equalize monochrome? it just prints 2 lists of values
        mono_incr = 12.8
        rgb_incr = (3*255)/10.0
        grey = [0]
        brightness = [0]
        for lev in range(1,10) :
            grey.append(lev*mono_incr)
            brightness.append(lev*rgb_incr)
            print(grey[-1], brightness[-1])
    
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
                    
if __name__ == '__main__' :
    TEST('EQM')

