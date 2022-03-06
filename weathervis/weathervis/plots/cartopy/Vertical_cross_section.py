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


H = [24122.6894480669, 20139.2203688489,17982.7817599549, 16441.7123200128,
    15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
    11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
    9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
    7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
    6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
    4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
    3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
    2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,
    1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
    1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
    952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
    630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
    396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
    230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
    108.885366240509, 83.2097562375566,58.7032686584901, 34.9801888163106,
    11.6284723290378]
# constants
t_zero = 273.15  # 0°C in [K]
cp = 1005.  # specific heat of dry air, constant pressure [J K^-1 kg^-1]
cv = 718.  # sepcific heat of dry air, constant volume J K^-1 kg^-1
L = 2.501e6  # latent heat of vaporization at 0°C J kg^-1
R = 287.04  # gas constant of dry air J kg^-1 K^-1
Rv = 461.5  # gas constant for water vapor J kg^-1 K^-1
g = 9.81  # gravitational acceleration in m s^-2
rdv = R / Rv
kappa = (cp - cv) / cp



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
def wind_dir_for_cross(xwind,ywind, alpha=None):
    #source: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    #https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    #https://stackoverflow.com/questions/21484558/how-to-calculate-wind-direction-from-u-and-v-wind-components-in-r
    u = np.zeros(shape=np.shape(xwind))
    v = np.zeros(shape=np.shape(ywind))
    wdir = np.empty( shape=np.shape(xwind) )
    for k in range(0, np.shape(wdir)[0]):
        #websais:#wdir[t,k,:,:] =  alpha[:,:] + 90 - np.arctan2(ywind[t,k,:,:],xwind[t,k,:,:])
        #Me:
        a =  np.arctan2( ywind[k,:], xwind[k,:] )  #mathematical wind angle in modelgrid pointing with the wind
        #a = a * (a >= 0) + (a + 2 * np.pi) * (a < 0)
        #a =  np.mod(a,np.pi)
        b = a*180./np.pi + 180.  # mathematical wind angle pointing where the wind comes FROM
        c = 90. - b   # math coordinates(North is 90) to cardinal coordinates(North is 0).
        if alpha[0] !=None:
            wdir[k,:] =  c[:] - alpha[:] #add rotation of modelgrid(alpha).
            #wdir[t,k,:,:] = np.subtract(c%360, alpha%360)
        wdir = wdir % 360  # making sure is between 0 and 360 with Modulo
    return wdir
class VERT_CROSS():
    def __init__(self, model, domain_lonlat, date, steps, data=None, extra_coords=None, legend=None, info=None,
            extra_names=None, m_level=None):
        self.model = model
        self.date  = date
        self.steps = steps
        self.data  = data
        self.domain_lonlat = domain_lonlat
        self.extra_coords = extra_coords
        self.extra_names = extra_names
        self.param_pl = []
        self.param_ml = ["air_temperature_ml", "specific_humidity_ml", "x_wind_ml", "y_wind_ml",
                         "cloud_area_fraction_ml","upward_air_velocity_ml",
                         "mass_fraction_of_cloud_ice_in_air_ml",
                         "mass_fraction_of_cloud_condensed_water_in_air_ml",
                         "mass_fraction_of_graupel_in_air_ml",
                         "mass_fraction_of_snow_in_air_ml",
                         "mass_fraction_of_rain_in_air_ml"]
        self.param_sfc = ["surface_air_pressure", "air_pressure_at_sea_level","atmosphere_boundary_layer_thickness",
                          "surface_geopotential"]
        self.param_sfx = []
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = None
        self.m_level = m_level
        self.mbrs = None
        self.url = None
        date = str(date)

    def retrieve_handler(self):

        print("\n######## Checking if your request is possible ############")
        self.param = self.param_pl + self.param_ml + self.param_sfc + self.param_sfx
        dmet,data_domain,bad_param = checkget_data_handler(all_param=self.param, date=self.date, model=self.model, step=self.steps,
                                     p_level=self.p_level, m_level=self.m_level,mbrs=self.mbrs,domain_lonlat=self.domain_lonlat,
                                     use_latest=True)

        self.dmet = dmet
        self.data_domain = data_domain
        print("DATA RETRIEVED")

        return dmet, data_domain,bad_param




    def crosssection_as_meteogramm(self,dirName,timest,extralines=None,kx=None,
                                   extrst=None,extrend=None):
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
        H = [24122.6894480669, 20139.2203688489,17982.7817599549, 16441.7123200128,
             15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
             11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
             9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
             7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
             6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
             4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
             3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
             2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,
             1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
             1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
             952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
             630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
             396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
             230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
             108.885366240509, 83.2097562375566,58.7032686584901, 34.9801888163106,
             11.6284723290378]
        
        print(self.dmet.atmosphere_boundary_layer_thickness.shape)
        TT  = self.dmet.air_temperature_ml[0,:,:,:]
        CAF = self.dmet.cloud_area_fraction_ml[0,:,:,:]
        Upd = self.dmet.upward_air_velocity_ml[0,:,:,:]
        RR  = self.dmet.mass_fraction_of_rain_in_air_ml[0,:,:,:]
        SN  = self.dmet.mass_fraction_of_snow_in_air_ml[0,:,:,:]
        GR  = self.dmet.mass_fraction_of_graupel_in_air_ml[0,:,:,:]
        CW  = self.dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[0,:,:,:]
        CI  = self.dmet.mass_fraction_of_cloud_ice_in_air_ml[0,:,:,:]
        QQ  = self.dmet.specific_humidity_ml[0,:,:,:]
        BLH = self.dmet.atmosphere_boundary_layer_thickness[0,0,:,:]
        UU = self.dmet.x_wind_ml[0,:,:,:]
        VV = self.dmet.y_wind_ml[0,:,:,:]
        alpha = self.dmet.alpha
        lats = self.dmet.latitude
        lons = self.dmet.longitude
        rlat = self.dmet.y
        rlon = self.dmet.x
        print('Point 1') 
        #self.dmet.u,self.dmet.v = xwind2uwind(self.dmet.x_wind_ml, self.dmet.y_wind_ml, self.dmet.alpha)
        #WS = wind_speed(self.dmet.x_wind_ml, self.dmet.y_wind_ml)
        WS = np.sqrt(UU**2+VV**2)
        print('Point 2') 
        Prec =(RR + SN +GR)*1000
        zi = {}
        # calculate the crosssection  
        P = ml2pl(self.dmet.ap, self.dmet.b, self.dmet.surface_air_pressure)
        P = P[0,:,:,:]
        print(P.shape)
        print('Point 3') 
        gph = self.dmet.surface_geopotential[0,0,:,:]
        re=6.3781*10**6
        hsurf=re*gph/(g*re-gph)
        dimz = np.shape(TT)[0]
        HH = np.zeros(np.shape(TT))
        for i,z in enumerate(H[65-dimz:65]):
            HH[i,:,:] = z
        # give the height interval that the field is goid to be interpolated to. currently lowest 5000 m in 100m steps
        heights=np.arange(0,5000,100)
        # crosssections: first Ny-Ålesund - Kiruna, second Kiruna-Andernes-Zone, third Kiruna-Sodankylä-Russia
        coos2 = [((12.1,78.93),(20.18,67.50)),((4.41,70.55),(20.18,67.50)),
                 ((20.18,67.50),(29.268,67.102))] 
        K=[1,0,0]
        stnam = ['Ny-Ålesund','Zone end','Kiruna']
        endnam   = ['Kiruna','Kiruna','Russia']
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
        clat = -23.6
        plon = -25.0
        plat  = 77.5
        fignam = ['NyA-Kir','Zone-Kir','Kir-Rus']
        print('Point 4')
        #Compute cross section
        for cc,kk,st,ed,fnam in zip(coos2,K,stnam,endnam,fignam):
            cross = CrossSection({'CAF':CAF,'CW':CW,'CI':CI,'QQ':QQ,'BLH':BLH,'WS':WS,
                                  'PREC':Prec,'Upd':Upd,'hsurf':hsurf,'lat':lats,
                                  'rlat':rlat,'lon':lons,'rlon':rlon,'p':P/100.,
                                  'z':HH+hsurf,'UU':UU,'VV':VV,'alpha':alpha,
                                  'TT':TT},
                                 cc,heights,version='rotated',pollon=plon,
                                 pollat=plat,flip=True,int2z=True,model=model) #polgam=180,
            #x,zi  = np.meshgrid(cross.lat,cross.pressure)
            print('Point 5, '+st)
            if kk==0:x,zi  = np.meshgrid(cross.lon,cross.pressure)
            else:x,zi  = np.meshgrid(cross.lat,cross.pressure)
    
            data={}
            data['CAF']  = cross.CAF
            data['CW']   = cross.CW
            data['CI']   = cross.CI
            data['QQ']   = cross.QQ
            data['BLH']  = cross.BLH+cross.hsurf
            data['WS']   = cross.WS
            data['PREC'] = cross.PREC
            data['Upd']  = cross.Upd
            data['TT']   = cross.TT
            # for wdir
            data['UU']    = cross.UU
            data['VV']    = cross.VV
            data['alpha'] = cross.alpha
            print('Point 6, '+st)
            wdir = wind_dir_for_cross(data['UU'],data['VV'],data['alpha'])
            u,v  = windfromspeed_dir(data['WS'],wdir)
            
            # the interpolated lapse rate somehwo alsways gives nan
            dtdz2 = (data['TT'][1::,:]-data['TT'][0:-1,:]) *10 # C/ km, *10 due to the interpolated height intervals of 100m
            # for the CAF hashing
            minCAF = np.zeros(np.shape(data['CAF']))
            xx,yy= np.where(data['CAF']<0.5)
            minCAF[xx,yy] = 1
            xx,yy= np.where(data['CAF']<0.01)
            minCAF[xx,yy] = 0
            xx,yy= np.where(data['CAF']>0.5)
            minCAF[xx,yy] = 2
            minCAF[np.where(minCAF==0)] = np.nan
            maxCAF = np.zeros(np.shape(data['CAF']))
            xx,yy= np.where(data['CAF']>0.5)
            maxCAF[xx,yy] = 1
            maxCAF[np.where(maxCAF==0)] = np.nan
            
            prec =data['PREC'].copy()
            prec[np.where(prec<0.05)] = np.nan
            
            Upd2 = data['Upd'].copy()
            Upd2[np.where(np.abs(Upd2)<0.15)] = np.nan
            
            # plot
            
            # first the humidity and wind
            fig = plt.figure(figsize=(14,12))
            # define variables for subplots
            gs = gridspec.GridSpec(2,1)
            
            ax2 = fig.add_subplot(gs[0,0])
            ax2.set_facecolor('grey')
            ax2.set_ylabel('Altitude [m]',fontsize=15)
            ax3 = fig.add_subplot(gs[1,0])
            ax3.set_facecolor('grey')
            ax3.set_ylabel('Altitude [m]',fontsize=15)
            aa = [ax2,ax3]
            # the cloud area fraction and wind speed
            levels = np.linspace(0.0, 3, 21)
            pc = ax2.contourf(x,zi,data['QQ']*1000,levels,cmap='gnuplot2_r',
                              extend='both')
            cs1 = ax2.contour(x,zi,data['CW']*1000,
                              list(np.linspace(0.01,np.nanmax(data['CW']*1000),5)),
                              colors='red',linewidths=2)
            cs2 = ax2.contour(x,zi,data['CI']*1000,
                              list(np.linspace(0.001,np.nanmax(data['CI']*1000),5)),
                              colors='cyan',linewidths=2)
            custom_lines = [Line2D([0], [0], color='red', lw=4),
                            Line2D([0], [0], color='cyan', lw=4)]
            ax2.legend(custom_lines,['Cloud water','Cloud ice'],fontsize=14,
                                     edgecolor='w',loc='upper left')
            ax2.set_ylim(ymin=0,ymax=5000)
            ax2.text(x[-1,0],-520,st,fontsize=15)
            ax2.text(x[1,-50],-520,ed,fontsize=15)
            cbar_ax = fig.add_axes([0.91, 0.563, 0.015, 0.3])
            cbar = fig.colorbar(pc, cax=cbar_ax,\
                                   orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel('Spec. Hum. [g/kg]',fontsize=15)
            ax2.invert_xaxis()
            levels = np.linspace(0.0, 21, 15)
            pw = ax3.contourf(x,zi,data['WS'],levels,cmap='RdYlBu_r',
                              extend='both')
            ax3.barbs(x[::6,::60], zi[::6,::60], u[::6,::60] * 1.943844,
                      v[::6,::60] * 1.943844, length=7, zorder=1000,alpha=0.5,
                      sizes=dict(emptybarb=0.25, spacing=0.15, height=0.4))
    
            ax3.invert_xaxis()
            cbar_ax = fig.add_axes([0.91, 0.563-0.413, 0.015, 0.3])
            cbar = fig.colorbar(pw, cax=cbar_ax,\
                                   orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel('wind speed [m/s]',fontsize=15) 
            ax3.set_ylim(ymin=0,ymax=5000)
            ax3.text(x[-1,0],-520,st,fontsize=15)
            ax3.text(x[1,-50],-520,ed,fontsize=15)
            # whan orinting on lon go from west to east
            if kk==0:
                for a in aa:
                    a.invert_xaxis()

            plt.savefig(dirName +"VCS_wind_"+ fnam+ "_step"+str(timest)+".png", bbox_inches = "tight", dpi = 200)
            print('Saved crosssection as {}'.format(dirName +"VCS_wind_"+ fnam+ "_step"+str(timest)+".png"))

            plt.clf()
            plt.close()
                    
            # second the lapse rate and cloud
            fig = plt.figure(figsize=(14,12))
            # define variables for subplots
            gs = gridspec.GridSpec(2,1)
            
            ax2 = fig.add_subplot(gs[0,0])
            ax2.set_facecolor('grey')
            ax3 = fig.add_subplot(gs[1,0])
            ax3.set_facecolor('grey')
            aa = [ax2,ax3]
            # the cloud area fraction and wind speed
            levels = np.linspace(0.0, 21.0, 7)
            pc = ax2.pcolormesh(x,zi,dtdz2,vmin=-10,vmax=10,cmap='RdYlBu_r')
            #levels = np.linspace(0.0, 0.49, 7)
            rr = ax2.contourf(x,zi,prec,cmap='cool',
                              extend='both')
            cf1 = ax2.contourf(x,zi,minCAF,colors='none',hatches=['--','---'])
            artists, labels = cf1.legend_elements(str_format='{:2.1f}'.format)
            ax2.legend(artists, ['1-50% Cloud cover','50-100% Cloud cover'],
                       handleheight=2,framealpha=0.3,loc='upper left',fontsize=14)
            ax2.plot(x[-1,:],data['BLH'],linewidth=3,color='k')
            ax2.set_ylim(ymin=0,ymax=5000)
            ax2.text(x[-1,0],-520,st,fontsize=15)
            ax2.text(x[1,-50],-520,ed,fontsize=15)
            cbar_ax = fig.add_axes([0.91, 0.563, 0.015, 0.3])
            cbar = fig.colorbar(pc, cax=cbar_ax,\
                                   orientation='vertical',extend='both')
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel('lapse rate [°C/km]',fontsize=15)
            cbar_rr = fig.add_axes([0.7, 0.845, 0.18, 0.015])
            cbar2 = fig.colorbar(rr, cax=cbar_rr,\
                                   orientation='horizontal')
            cbar2.ax.set_facecolor('none')
            cbar2.ax.tick_params(labelsize=12)
            cbar2.ax.set_xlabel('Precip [g kg$^{-1}$]',fontsize=13)
            ax2.invert_xaxis()
    
            pw = ax3.pcolormesh(x,zi,data['CAF'],cmap='Blues')
            cs1 = ax3.contour(x,zi,data['TT']-273.15,[-40,-30,-20,-15,-10,-5,0,5],
                              colors='orange',linewidths=2,linestyles='solid')
            #cs2 = ax3.contour(x,zi,data['Upd'],[-1.5,-1,-0.5,-0.2,0.2,0.5,1,1.5],
             #                 colors='blueviolet',linewidths=2,linestyles='solid')
            cs2 = ax3.contourf(x,zi,Upd2,[-1.5,-1,-0.5,-0.2,0.2,0.5,1,1.5],
                              cmap='PRGn',alpha=0.7,extend='both')
            custom_lines = [Line2D([0], [0], color='orange', lw=4)]#,
                            #Line2D([0], [0], color='blueviolet', lw=4)]
            ax3.legend(custom_lines,['temperature [°C]'],fontsize=14,
                                     edgecolor='w',loc='upper left') #,'vertical velocity [m/s]'
            ax3.clabel(cs1, inline=1,fmt='%.0f', fontsize = 14)
            #ax3.clabel(cs2, inline=1,fmt='%.1f', fontsize = 14)
            cbar_ax = fig.add_axes([0.91, 0.563-0.413, 0.015, 0.3])
            cbar = fig.colorbar(pw, cax=cbar_ax,\
                                   orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel('cloud area fraction',fontsize=15) 
            cbarW = fig.add_axes([0.7, 0.845-0.4, 0.18, 0.015])
            cbarW = fig.colorbar(cs2, cax=cbarW,\
                                   orientation='horizontal')
            cbarW.ax.set_facecolor('none')
            cbarW.ax.tick_params(labelsize=11)
            cbarW.ax.set_xlabel('vert. wind speed [m s$^{-1}$]',fontsize=13)
            ax3.invert_xaxis()
            ax3.text(x[-1,0],-520,st,fontsize=15)
            ax3.text(x[1,-50],-520,ed,fontsize=15)
            # whan orinting on lon go from west to east
            if kk==0:
                for a in aa:
                    a.invert_xaxis()
     
            plt.savefig(dirName +"VCS_cloud_"+ fnam+"_step"+str(timest)+".png", bbox_inches = "tight", dpi = 200)
            print('Saved crosssection as {}'.format(dirName +"VCS_cloud_"+ fnam+ "_step"+str(timest)+".png"))

if __name__ == "__main__":
    import argparse
    def none_or_str(value):
        if value == 'None':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_lonlat", default=[-4.770555, 41.700291, 64.6152131, 79.477055], help="[ lonmin, lonmax, latmin, latmax]")
    parser.add_argument("--domain_name", default="cross_region", help="no help for you")
    parser.add_argument("--extra_names", default=None, help="give list of names for additional crossection")
    parser.add_argument("--extra_coords", default=None, help="give list of extra coordinates")
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    parser.add_argument("--id", default=None, help="Display legend", type=str)
    parser.add_argument("--outpath", default=None, help="Display legend", type=str)
    parser.add_argument("--m_level", default=[0,64], nargs="+", type=int, help="model levels to retrieve --m_level 30 64 gives lowest 35 model levels")

    args = parser.parse_args()
    print(args.steps)
    for dt in args.datetime:
        dirName = setup_met_directory(dt, runid=args.id, outpath=args.outpath)

        for st in range(args.steps[-1]): # for now assuming that we start at 0

            VC = VERT_CROSS(date=dt, steps=st, model=args.model, domain_lonlat = args.domain_lonlat,
                            extra_names = args.extra_names,
                            extra_coords = args.extra_coords,
                            info=args.info, m_level=args.m_level)
            VC.retrieve_handler()


            VC.crosssection_as_meteogramm(dirName=dirName,timest=st)

