import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import pandas as pd
import shapely as sh
import numpy as np
import geog as gg

def add_ISLAS_overlays(ax,col='red'):

  lon=[10, 0, 0, 5,10.5,20.1,22,29,29.5,10]
  lat=[79,72,60,57,58.5,58.5,65,65,79  ,79]
  # Coordinates of boundary points of the flight area : 
  #1  :  N72째 - E000�2  :  N60째 - E000째3  :  N57째 - E005째
  #4  :  N58째3 E010째3	5  :  N58째3 E020�6  :  N65째 - E022째
  #7  :  N65째 - E029째	8  :  N79째 - E0299  :  N79째 - E010째

  #GND-2500FT
  lat_f1 = [69.43, 69.45, 69.05, 69.27, 69.33, 68.75, 67.9, 67.9, 
          68.18, 68.28, 68.3, 68.44, 68.44, 69.23, 69.5, 69.5, 
          69.67, 69.5, 69.5, 69.48, 69.42, 69.4, 69.43]
  lon_f1 = [19.5, 20.0, 20.0, 18.75, 18.45, 17.02, 16.75, 16.75, 
          14.9, 14.95, 14.58, 14.07, 14.07, 14.55, 16.67, 16.67, 
          17.7, 18.17, 18.48, 18.4, 19.05, 19.03, 19.5]
  #GND-FL115 (11500ft)
  lat_f2 = [69.33, 69.27, 69.05, 69.033, 68.73, 68.7, 68.7, 68.3, 
          68.6, 68.55, 68.52, 68.1, 68.2, 67.9, 68.37, 68.75, 69.33]
  lon_f2 = [18.45, 18.75, 20.0, 20.05, 20.27, 20.22, 20.21, 20.0, 
          18.4, 18.1, 18.11, 18.11, 17.41, 16.75, 16.92, 17.06, 18.45]

  # plot domain outline
  with ax.hold_limits():
    ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())

  # plot CR22 outline
  with ax.hold_limits():
    ax.plot(lon_f1,lat_f1,linewidth=1.0,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_f2,lat_f2,linewidth=1.0,color=col,linestyle='solid',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../../data/sites.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=30,color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../../data/airports.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=100,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

  # add range circle for Kiruna
  p = sh.geometry.Point([20.31891,67.8222]) # location
  n_points = 50
  angles = np.linspace(0, 360, n_points)
  d = 300 * 1000 * 1.852  # nautical miles
  polygon1 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  d = 450 * 1000 * 1.852  # nautical miles
  polygon2 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  with ax.hold_limits():
    ax.plot(polygon1[:,0],polygon1[:,1],linewidth=1.0,color=col,linestyle='dashdot',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(polygon2[:,0],polygon2[:,1],linewidth=1.0,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())

