#!/opt/local/bin/python2.7

import numpy as np
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from healpy.visufunc import projtext
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from numpy import genfromtxt
from matplotlib import cm


grey_cmap = cm.Greys
grey_cmap.set_under("w") # sets background to white

orange_cmap = cm.Oranges
orange_cmap.set_under("w") # sets background to white

map = hp.read_map('sky.fits')
#Smoothes the map with a 1-degree FWHM Gaussian (fwhm given in radians).
map_smth = hp.sphtfunc.smoothing(map, fwhm = 0.017, iter = 1)


#hp.mollview(map_smth,title=" ", rot=[0,0,0], flip='astro', min=0, cbar=False, cmap=orange_cmap)
hp.mollview(map_smth,title=" ", rot=[-180,0,0], flip='astro', min=0, cbar=False, cmap=orange_cmap)
hp.graticule(dpar=15,dmer=30)
for i in range(2,24,2):
    text = hp.projtext( i*180/12+5, 4,  str(i)+'h', lonlat=True, coord='G',
                       fontsize = 'medium', fontweight = 100, zorder = 1)
    text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='w')])
for i in range(-75,0,15):
    text = hp.projtext( 360,i+7*i/75,   str(i)+r'$^\circ$', lonlat=True, coord='G',
                       fontsize = 'medium', fontweight = 100, zorder = 10)
    text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='w')])
for i in range(0,90,15):
    text = hp.projtext( 360,i-3*i/75,   str(i)+r'$^\circ$', lonlat=True, coord='G',
                       fontsize = 'medium', fontweight = 100, zorder = 10)
    text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='w')])
    
    
#projview(map_smth,coord=["E"],rot=[-180,0,0],graticule=True,rot_graticule=True,longitude_grid_spacing=30latitude_grid_spacing=30,graticule_labels=True,xlabel="longitude",ylabel="latitude",cbar=False,cmap=orange_cmap,projection_type="mollweide",);

plt.savefig('sky.png')
