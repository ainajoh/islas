#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:05:13 2022

@author: marvink
"""
# %%
from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.check_data import *

from weathervis.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import matplotlib.patheffects as pe
from matplotlib import gridspec
from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
from copy import deepcopy
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
from weathervis.checkget_data_handler import *
sys.path.insert(0, '/home/centos/plots/Marvin/')
from small_tools_meps import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)





def setup_met_directory(modelrun, extra_names=None, runid=None, outpath=None):
    global OUTPUTPATH
    if outpath != None:
        OUTPUTPATH=outpath
    if runid !=None:
        projectpath = setup_directory( OUTPUTPATH, "{0}-{1}".format(modelrun,runid) )
    else:
        projectpath = setup_directory( OUTPUTPATH, "{0}".format(modelrun) )

    dirName = projectpath + "/"

    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName

# tools               
def rotate_points_AROME(pole_longitude, pole_latitude, lon, lat, direction='n2r',model='AromeArctic'):
    """Rotate lon, lat from/to a rotated system    Parameters
    ----------
    pole_longitude: float
        longitudinal coordinate of the rotated pole
    pole_latitude: float
        latitudinal coordinate of the rotated pole
    lon: array (1d)
        longitudinal coordinates to rotate
    lat: array (1d)
        latitudinal coordinates to rotate
    direction: string, optional
        direction of the rotation;
        n2r: from non-rotated to rotated (default)
        r2n: from rotated to non-rotated
    model: string, optional
    	used model    Returns
    -------
    rlon: array
    rlat: array
    """
    lon = np.array(lon)
    lat = np.array(lat)
    globe = ccrs.Globe(semimajor_axis=6371000.)
    if model=='MEPS':
    	rotatedgrid = ccrs.LambertConformal(central_longitude=pole_longitude,
                                               central_latitude=pole_latitude,
                                               standard_parallels=(63.3,63.3),
                                               globe=globe)
    elif model=='AromeArctic':
    	rotatedgrid = ccrs.LambertConformal(central_longitude=pole_longitude,
                                               central_latitude=pole_latitude,
                                               standard_parallels=(77.5,77.5),
                                               globe=globe)
    standard_grid = ccrs.Geodetic()

    if direction == 'n2r':
        rotated_points = rotatedgrid.transform_points(standard_grid, lon, lat)
    elif direction == 'r2n':
        rotated_points = standard_grid.transform_points(rotatedgrid, lon, lat)
    rlon, rlat, _ = rotated_points.T
    return rlon, rlat


class VERT_CROSS_flexpart():
    def __init__(self, model, domain_lonlat, date, steps, data=None,
            extra_coords=None, legend=None, info=None,
            extra_names=None):
        self.model = model
        self.date  = date
        self.steps = steps
        self.data  = data
        self.domain_lonlat = domain_lonlat
        self.extra_coords = extra_coords
        self.extra_names = extra_names
        self.param_pl = []
        self.param_ml = ["rel1","rel2","rel3","rel4","rel5"]
        self.param_sfc= []
        self.param_sfx = []
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        date = str(date)


    def vert_crosssection(self,dat,date,dirName,timest,extralines=None,kx=None,
                                   extrst=None,extrend=None,st_nam="NYA",
                                   end_nam="KRN",coords=[12.1,20.18,67.50,78.93],orient=0):
        """create cross section 
         ----------
        timest:        
                    individual step
        extralines: array, optional
                    additional cross section in the form ((lon1,lat1),(lon2,lat2))
        kx    : str
                x-axis of extra cross section. if kx = 'lon' longitudes on x 
                                               if kx = 'lat' latitudes on x
        extrst : str
                name of start location
        extrend: str
                name of end location
    
         """
        # give the height interval that the field is good to be interpolated to. 
        # currently lowest 5000 m in 100m steps
        heights=np.arange(0,5000,100)
        # crosssections: first Ny-Aesund - Kiruna, second Russian border, fliht at 2503
        # yes I know that this is a more than strange way of "realising" these sections
        coos2=[((coords[0],coords[2]),(coords[1],coords[3])),((30,80),(30,67)),
                ((7,73.5),(12,73.5))]
        K=[orient,0,1]
        stnam=[st_nam,'RusSval','Flight4_st']
        endnam=[end_nam,'RusFin','Flight4_end']
        # in case of extra_lines
        if kx == 'lon':
            coos2.append(extralines)
            K.append(0)
            stnam.append(extrst)
            endnam.append(extrend)
        elif kx == 'lat':
            coos2.append(extralines)
            K.append(1)        
            stnam.append(extrst)
            endnam.append(extrend)

        # for now just for model == 'AromeArctic':
        model = 'AromeArctic'
        clat  = -23.6
        plon  = -25.0
        plat  = 77.5
        lons  = dat.variables['XLONG'][:,:]
        lats  = dat.variables['XLAT'][:,:]
        rlon, rlat = rotate_points_AROME(plon,plat,lons,lats,'n2r',
                                         model='AromeArctic')
        fignam = []
        for st,ed in zip(stnam,endnam):
            fignam.extend([st+'-'+ed])

        xxx = np.shape(lons)[0]
        yyy = np.shape(lons)[1]
        zzz = np.shape(dat.variables['ZTOP'])[0]
        HH  = np.zeros([zzz,xxx,yyy])
        for i,z in enumerate(dat.variables['ZTOP'][:]):
            HH[i,:,:] = z
        print('Point 4')
        #Compute cross section
        for cc,kk,st,ed,fnam in zip(coos2,K,stnam,endnam,fignam):
            # assuming five releases
            rel1 = dat.variables['CONC'][timest,0,:,:,:]
            rel2 = dat.variables['CONC'][timest,1,:,:,:]
            rel3 = dat.variables['CONC'][timest,2,:,:,:]
            rel4 = dat.variables['CONC'][timest,3,:,:,:]
            rel5 = dat.variables['CONC'][timest,4,:,:,:]
            #cross = CrossSection(dmet,
            cross = CrossSection({'rel1':rel1,'rel2':rel2,'rel3':rel3,'rel4':rel4,
                                  'rel5':rel5,'lat':lats,
                                   'rlat':rlat,'lon':lons,'rlon':rlon,
                                   'z':HH},
                                   cc,heights,version='rotated',pollon=plon,
                                   pollat=plat,flip=False,int2z=True,model=model) #polgam=180,
            #x,zi  = np.meshgrid(cross.lat,cross.pressure)
            print('Point 5, '+st)
            if kk==0:
                x,zi  = np.meshgrid(cross.lon,cross.pressure)
            else:
                x,zi  = np.meshgrid(cross.lat,cross.pressure)
   
            data={}
            data['rel1'] = cross.rel1
            data['rel2'] = cross.rel2
            data['rel3'] = cross.rel3
            data['rel4'] = cross.rel4
            data['rel5'] = cross.rel5
           

            cma = [plt.cm.Greys,plt.cm.Greens,plt.cm.Blues,plt.cm.Reds,plt.cm.Purples]
            nam = ['rel1','rel2','rel3','rel4','rel5']
            offs = [0.16,0.31,0.46,0.61,0.76]
            gg ={}
            fig = plt.figure(figsize=(14,6))
            ax = plt.gca()
            for n,cc in zip(nam,cma):
                conc = data[n].copy()
                conc[np.where(conc==0)] = np.nan
                gg[n] = plt.pcolormesh(x,zi,conc,cmap=cc,vmin=0,vmax=1,alpha=1)
            for n,of in zip(nam,offs):
                cbar_gg = fig.add_axes([of, 0.845, 0.12, 0.015])
                cbar2 = fig.colorbar(gg[n], cax=cbar_gg,\
                                       orientation='horizontal',
                                       label = n)
            if orient ==0:
                ax.invert_xaxis()


            plt.savefig(dirName +"VCS_flexpart_"+fnam+"_{0}+{1:02d}.png".format(date,timest),
                    bbox_inches = "tight", dpi = 200)
            print('Saved flexpart crosssection as {}'.format(dirName +"VCS_flexpart_"+ fnam+ "_{0}+{1:02d}.png".format(date,timest)))

            plt.clf()
            plt.close()
                    

if __name__ == "__main__":
    import argparse
    def none_or_str(value):
        if value == 'None':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 12 gives time 0 to 12 in steps of 6")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_lonlat", default=[-4.770555, 41.700291, 64.6152131, 79.477055], nargs="+", help="[ lonmin, lonmax, latmin, latmax]", type=float)
    parser.add_argument("--points_lonlat", default=[-4.770555, 41.700291, 64.6152131, 79.477055], nargs="+", help="[ lonmin, lonmax, latmin, latmax]", type=float)
    parser.add_argument("--domain_name", default="cross_region", help="no help for you")
    parser.add_argument("--extra_names", default=None, help="give list of names for additional crossection")
    parser.add_argument("--extra_coords", default=None, help="give list of extra coordinates")
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    parser.add_argument("--id", default=None, help="Display legend", type=str)
    parser.add_argument("--orient", default=None, help="orientation", type=int)
    parser.add_argument("--outpath", default=None, help="Display legend", type=str)
    parser.add_argument("--start_name",default="NYA",help="name of start location", type=str)
    parser.add_argument("--end_name",default="KRN",help="name of end location", type=str)

    args = parser.parse_args()
    print(args.steps)
    for dt in args.datetime:
        dirName = setup_met_directory(dt, runid=args.id, outpath=args.outpath)
        # load in flexpart data using hard coded path, can be changed later
        dat = Dataset("/home/centos/flexpart/fp_arome/fp_arome_"+dt+"_forecast_S1.nc")
        for st in np.arange(args.steps[0],args.steps[1],3): # can be changed to less or more steps

            VC = VERT_CROSS_flexpart(date=dt, steps=st, model=args.model,
                                     domain_lonlat = args.domain_lonlat,
                                     extra_names = args.extra_names,
                                     extra_coords = args.extra_coords,
                                     info=args.info)

            VC.vert_crosssection(dat,date=dt,dirName=dirName,timest=st,end_nam=args.end_name,
                                 st_nam=args.start_name,
                                 coords=args.points_lonlat,orient=args.orient)

