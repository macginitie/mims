# mims
MacGinitie's Image Munging Stuff

(from https://dictionary.cambridge.org/us/dictionary/english/munging:
"...the process of changing data into another format \[...\] so that it can be used or processed...")

I could've called it "Image Processing" but then my acronymic would've honored a RISC chip maker instead of the great Forrest M. Mims III -- or maybe I just like the word "munging"...


picmerj.py
==========

a Python 3.x script that uses PIL to "merge" two pictures, creating a third.

fairly rudimentary, but it can produce some interesting (at least to the author) images.
it does a pixel-by-pixel comparison of the images, and chooses one of the input pixels
to write to the output image, or else combines them in some way, based on a commandline
flag & maybe one or more parameters (e.g., an (r,g,b) triplet).

it tries to write the output file to a subfolder named "newimages"; if it fails, it tries
to create "default.jpg" in the folder where you launched it.

example usage:

python picmerj.py image1.jpg image2.png b

will produce newimages/image1-image2-b.jpg in which the "bluer" pixel of each input image
will be in the output image, where "bluer" is simply the r,g,b tuple with a higher value 
for b (often not what a human eye would see as "bluer").


mip.py
======

MacGinitie's Image Processor

a Python 3.x script that reads an input image and produces a new output image, applying
some kind of image processing (or munging if you prefer) specified by one or more commandline parameters.
