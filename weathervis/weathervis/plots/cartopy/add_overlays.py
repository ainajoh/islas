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

  # cross-section coordinates
  lon_s1=[12.1,12.23520606,12.65944174,13.21400478,13.58760356,13.94094611,14.33871462,
      14.65491706,15.03016339,15.37328566,15.64192952,15.89794657,16.19885511,
      16.4963865 ,16.71676549,16.98148237,17.18253143,17.43431235,17.66910028,
      17.84448999,18.01302922,18.27786088,18.43192251,18.58029156,18.76995314,
      18.95664375,19.08803228,19.25955956,19.42811494,19.54508722,19.70103341,
      19.85399797,19.95863716,20.06000296,20.24056098,20.18]
  lat_s1=[78.93,78.9201046 ,78.57888799,78.24135548,77.89877258,77.5557036 ,77.22982397,
      76.88585191,76.5273737 ,76.20003903,75.85499837,75.50972759,75.18141804,
      74.82113513,74.47523171,74.14616744,73.80000724,73.43894861,73.10939372,
      72.763008  ,72.4166229 ,72.07180271,71.72538868,71.37903104,71.04917438,
      70.68772372,70.3415569 ,70.01178108,69.6505138 ,69.3046819 ,68.97513427,
      68.61420174,68.26883362,67.92366473,67.57918425,67.5]
  lon_s2=[20.18,20.28568132,20.49090086,20.73650784,21.02311294,21.26856487,21.47283874,
      21.71810188,21.96334317,22.2486139 ,22.45183297,22.69661679,22.9413448,
      23.2252967 ,23.46970516,23.67152584,23.91560771,24.19809262,24.44178398,
      24.68535977,24.88561967,25.12879474,25.4093394 ,25.65203706,25.85088216,
      26.09312171,26.37188085,26.61357582,26.85508309,27.05209295,27.29307225,
      27.56950503,27.80985806,28.04998496,28.24501692,28.27]
  lat_s2=[67.5,67.46953351,67.45478106,67.45554947,67.44025305,67.44016759,67.42410167,
      67.42329479,67.42209443,67.40450638,67.38713581,67.38475896,67.38198961,
      67.36258133,67.35896473,67.33997587,67.33564455,67.31442918,67.30925487,
      67.30369047,67.28309965,67.27682545,67.2533688 ,67.24625761,67.22440035,
      67.21658417,67.19135463,67.18270726,67.17367437,67.15024994,67.14051858,
      67.11309344,67.10253877,67.09160174,67.06663179,67.1]
  lon_s3=[4.41, 4.44953925 ,5.0407378  ,5.53749556 ,6.08798687 ,6.5772435 , 7.11912821,
      7.63533681, 8.16884073, 8.64304204, 9.16768893, 9.68759595,10.18598354,
     10.69721632,11.15171019,11.65391332,12.10042551,12.63114747,13.06987576,
     13.55419437,13.98490735,14.49862724,14.92151709,15.38792679,15.80283199,
     16.29943515,16.75224481,17.15515267,17.59904567,18.03386792,18.46903825,
     18.8563932 ,19.2828046 ,19.70266439,20.12049871,20.18]
  lat_s3=[70.55,70.51902294,70.46049417,70.3917416 ,70.31032248,70.23864125,70.15399274,
     70.09841456,70.01050593,69.93307996,69.84210155,69.74953743,69.68640955,
     69.59078609,69.50653244,69.40804363,69.3212549 ,69.23784325,69.14848771,
     69.04439499,68.95264677,68.86334481,68.7691707 ,68.65979221,68.5633659,
     68.46848725,68.35525772,68.25542252,68.13984352,68.05444846,67.93648026,
     67.83245017,67.71228247,67.62237927,67.49997071,67.5]

  # plot domain outline
  with ax.hold_limits():
    ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())

  # plot CR22 outline
  with ax.hold_limits():
    ax.plot(lon_f1,lat_f1,linewidth=1.0,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_f2,lat_f2,linewidth=1.0,color=col,linestyle='solid',zorder=12,transform=ccrs.PlateCarree())

  # plot cross sections
  with ax.hold_limits():
    ax.plot(lon_s1,lat_s1,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_s2,lat_s2,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_s3,lat_s3,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())

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

  # add ship track
  ship="/home/centos/ship/marinetraffic_positions"
  locs = pd.read_csv(ship,sep=',',skipinitialspace=True)
  with ax.hold_limits():
    ax.plot(locs["LON"],locs["LAT"],'-',color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
 
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

