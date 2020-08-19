
from PIL import Image

img = Image.open('/pix/newer/statue.jpg')
img_pix = img.load()
(x_pixels, y_pixels) = img.size
width = x_pixels//2
print( x_pixels, y_pixels, width )

new_image = Image.new(img.mode, (width, y_pixels))
new_pixbuf = new_image.load()

for y in range(y_pixels):
    for x in range(width):
        new_pixbuf[x,y] = img_pix[x,y]
        
newfile = 'images/statue-left.png'
new_image.save(newfile)        

for y in range(y_pixels):
    for x in range(width):
        new_pixbuf[x,y] = img_pix[x + width,y]

newfile = 'images/statue-right.png'
new_image.save(newfile)
