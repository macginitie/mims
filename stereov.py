#!/usr/bin/env python
#
# sereov.py 
#

from PIL import Image
from PIL import ImageFilter
import sys
import random

white = (255,255,255)
black = (0,0,0)
max_red = (255,0,0)
max_green = (0,255,0)
max_blue = (0,0,255)

     
def clamp(val,min,max) :
    if (val < min) : return min
    if (val > max) : return max
    return val
               
def sum_channels(pixel) :
    return pixel[0] + pixel[1] + pixel[2]
    
def red_channel(pixel) :
    return pixel[0]
    
def green_channel(pixel) :
    return pixel[1]
    
def blue_channel(pixel) :
    return pixel[2]
    
def euclidean_dist(px,px2) :
    return (px[0] - px2[0])**2 + (px[1] - px2[1])**2 + (px[2] - px2[2])**2
    
def dist_to_target(px) :
    return euclidean_dist(px, (target_red, target_green, target_blue))
    
def average(px1, px2) :
    return (int((px1[0] + px2[0])/2), int((px1[1] + px2[1])/2), int((px1[2] + px2[2])/2))
   
def closer(px1, px2) :
    if (dist_to_target(px1) < dist_to_target(px2)) :
        return px1
    return px2
    
def further(px1, px2) :
    if (dist_to_target(px1) > dist_to_target(px2)) :
        return px1
    return px2
    
def diff_patch( im1, im2, x, y, patch_size ) :
    totaldiff = 0
    for i in range( x, x+patch_size ) :
        for j in range( y, y+patch_size ) :
            # 2DO: compare bounds, here or better yet in caller
            try :
                totaldiff += euclidean_dist( im1[i,j], im2[i,j] )
            except :
                pass
    return totaldiff // 1000

# the idea was to slide a [patch_size X patch_size] window horizontally
# and find where the "match" is closest -- i.e., where the difference, 
# as computed by diff_patch(), is least -- then use the lateral distance
# of that best match as a depth estimate (greater lateral distance would
# be interpreted as greater proximity to the "eyes").
# ...but that is not what this code is currently doing, as of 4/18/16
def estim8_depth( im1, im2, xcenter, ycenter, patch_size ) :
    lastdist = 999999999 # arbitrary big value
    dist_b4_last = lastdist + 1
    while lastdist < dist_b4_last :
        dist_b4_last = lastdist
        lastdist = diff_patch( im1, im2, xcenter, ycenter, patch_size )
    print( dist_b4_last )
    gs = clamp( dist_b4_last, 0, 255 )
    return ( gs, gs, gs )
    
# assumption: im2 is from the right eye, im1 the left
# also, patch_size is odd [i.e., (patch_size % 2) != 0]
def compare(im1, im2, width, height, depth_map, patch_size) :
    midpt = 1 + (patch_size // 2)
    for ycenter in range( 1 + midpt, height - (1 + midpt) ) :
        for xcenter in range( 1 + midpt, width - (1 + midpt) ) :
            depth_map[xcenter, ycenter] = estim8_depth( im1, im2, xcenter, ycenter, patch_size )
        
if __name__ == '__main__':

    try :
        print( 'trying to open ' + sys.argv[1] )
        img = Image.open(sys.argv[1])
        print( 'trying to open ' + sys.argv[2] )
        img2 = Image.open(sys.argv[2])
    except :
        print( 'failed!' )
        exit()

    img_pix = img.load()
    (x_pixels, y_pixels) = img.size
    img2_pix = img2.load()
    (width, height) = img2.size

    if (width != x_pixels) : 
        print( 'error: images have different widths (', x_pixels, '!=', width, ')' )
        exit()
    if (height > y_pixels) :
        print( 'error: images have different heights (', y_pixels, '!=', height, ')' )
        exit()

    patch_size = 7 # 49 pixels
    depth_map = Image.new(img.mode, (width - (patch_size // 2), height - (patch_size // 2)))
    compare( img_pix, img2_pix, width, height, depth_map.load(), patch_size )
    depth_map.save('depth.jpg')
    