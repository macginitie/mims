#!python3

import sys

def load_rgb_colors( filename ) :
    clist = []
    c = open( filename ).readlines()
    for l in c :
        (r,g,b) = l.strip().split(' ')
        clist.append( (r,g,b) )
    return clist
    
if __name__ == '__main__' :
    #try :
    colors = load_rgb_colors( sys.argv[1] )
    print( colors )
    #except :
    #    print( 'error... please specify name of color list file on cmdline' )
