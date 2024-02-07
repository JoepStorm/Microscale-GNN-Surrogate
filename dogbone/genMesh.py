import re
import os
import numpy as np
import random
import math

class Geo:
    def __init__ (self, name):
        self.file = open (name,'w')
        self.counter = 0

    def point (self,x,y):
        self.counter = self.counter+1
        self.file.write('Point(' + str(self.counter) + ') = {' + str(x) + ', ' + str(y) + ',0,' + str(le) + '};\n')
        return self.counter

    def line (self,p1,p2):
        self.counter = self.counter+1
        self.file.write('Line(' + str(self.counter) + ') = {' + str(p1) + ', ' + str(p2) + '};\n')
        return self.counter

    def circle(self,p1,p2,p3):
        self.counter = self.counter+1
        self.file.write('Circle(' + str(self.counter) + ') = {' + str(p1) + ', ' + str(p2) + ', ' + str(p3) + '};\n')
        return self.counter 

    def loop (self,lst):
        self.counter = self.counter+1
        self.file.write('Line Loop(' + str(self.counter) + ') = {')
        for i, point in enumerate(lst):
            if i > 0:
                self.file.write (',' + str(point)) 
            else:
                self.file.write (str(point)) 
        self.file.write('};\n')
        return self.counter

    def planesurf (self,lst):
        self.counter = self.counter+1
        self.file.write('Plane Surface(' + str(self.counter) + ') = {')
        for i, point in enumerate(lst):
            if i > 0:
                self.file.write (',' + str(point)) 
            else:
                self.file.write (str(point)) 
        self.file.write('};\n')
        return self.counter

    def physsurf (self,surfid):
        self.counter = self.counter+1
        self.file.write('Physical Surface(' + str(self.counter) + ') = {' + str(surfid) + '};\n')
        return self.counter

nh = 20   # number of holes
r  = 1.7 # hole radius

scale = 4.0
b  = scale
l  = 16.0 * scale
rr = 300.0 * scale

gx = l / 1.2
gy = b / 1.75

le = 1.0

seed = 11

random.seed ( seed )

geo = Geo ('mesh.geo')

lb = geo.point(-l, -b)
rb = geo.point( l, -b)
rt = geo.point( l,  b)
lt = geo.point(-l,  b)
ct = geo.point( 0, rr)
cb = geo.point( 0,-rr)

hxs = []
hys = []

mainloop = geo.loop ([geo.circle (lb,cb,rb), geo.line(rb,rt), geo.circle(rt,ct,lt), geo.line(lt,lb)])

circloops = []

for i in range(nh):
    hx = random.uniform ( -gx + r, gx - r )
    hy = random.uniform ( -gy + r, gy - r )

    if len(hxs) > 0:
        while any(math.sqrt((hx-x)*(hx-x)+(hy-y)*(hy-y)) < 2*r for x,y in zip(hxs,hys)):
            hx = random.uniform ( -gx + r, gx - r )
            hy = random.uniform ( -gy + r, gy - r )

    center = geo.point(hx  ,hy)
    left   = geo.point(hx-r,hy)
    right  = geo.point(hx+r,hy)

    half1 = geo.circle(left,center,right)
    half2 = geo.circle(right,center,left)

    circloops.append ( geo.loop([half1, half2]) )

    hxs.append ( hx )
    hys.append ( hy )

surf = geo.planesurf ([mainloop] + circloops)

geo.physsurf(surf)
   
