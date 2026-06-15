#!/opt/local/bin/python2.7

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def discrete_cmap(colors):
    cmap = ListedColormap(colors)
    cmap.set_bad('white')
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(np.arange(len(colors)+1)-0.5, cmap.N)
    return cmap, norm


def render_map(map_data, filename, colors, graticule):
    cmap, norm = discrete_cmap(colors)
    hp.mollview(map_data, title=" ", rot=[-180, 0, 0], flip='astro',
                min=0, max=len(colors)-1, cbar=False, cmap=cmap,
                norm=norm, badcolor='white', bgcolor='white',
                notext=(not graticule), xsize=1600)
    if graticule:
        hp.graticule(dpar=15, dmer=30, color='0.75')
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    for ax in fig.get_axes():
        ax.set_facecolor('white')
    plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=200)
    plt.close(fig)


def render_zoom(map_data, filename, colors, pixels):
    if len(pixels) == 0:
        return

    nside = hp.npix2nside(len(map_data))
    vec = np.array(hp.pix2vec(nside, pixels))
    center = np.mean(vec, axis=1)
    center /= np.sqrt(np.sum(center*center))
    theta, phi = hp.vec2ang(center)
    lon = np.asarray(np.degrees(phi)).item()
    lat = np.asarray(90.0-np.degrees(theta)).item()
    cmap, norm = discrete_cmap(colors)

    hp.gnomview(map_data, title=" ", rot=[lon, lat, 0.0], flip='astro',
                xsize=900, ysize=900, reso=1.0, min=0,
                max=len(colors)-1, cbar=False, cmap=cmap, norm=norm,
                badcolor='white', bgcolor='white', notext=True)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    for ax in fig.get_axes():
        ax.set_facecolor('white')
    plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=200)
    plt.close(fig)


boundary = hp.read_map('sky.fits')
render_map(boundary, 'sky90_boundary.png', ['white', 'black'], 1)

if os.path.exists('sky.dat') and os.path.exists('sky90_boundary.dat'):
    density = np.loadtxt('sky.dat')
    boundary_ascii = np.loadtxt('sky90_boundary.dat')
    npix = len(density)

    check_map = np.zeros(npix)
    if os.path.exists('sky90_region.dat'):
        region_ascii = np.loadtxt('sky90_region.dat')
        check_map[region_ascii > 0.0] = 1.0
    else:
        order = np.argsort(density)
        cumulative = 0.0
        k = 0
        while cumulative < 0.9 and k < npix:
            k += 1
            cumulative += density[order[npix-k]]
        check_map[order[npix-k:npix]] = 1.0
    check_map[boundary_ascii > 0.0] = 2.0
    render_map(check_map, 'sky90_boundary_check.png',
               ['white', '0.45', 'black'], 0)
    render_zoom(check_map, 'sky90_boundary_check_zoom.png',
                ['white', '0.45', 'black'], np.nonzero(check_map)[0])
