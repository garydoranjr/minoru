#!/usr/bin/env python
import minoru
from PIL import Image

left, right = minoru.capture('/dev/video1', '/dev/video0')

print left
print right

im = Image.fromarray(left)
im.save("left.png")
im = Image.fromarray(right)
im.save("right.png")
