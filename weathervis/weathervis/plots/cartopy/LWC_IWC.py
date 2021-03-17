# %%
# python LWC_IWC.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

from weathervis.config import *
from weathervis.utils import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)

def domain_input_handler(dt, model, domain_name, domain_lonlat, file):
  if domain_name or domain_lonlat:
    if domain_lonlat:
      print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
      data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
    else:
      data_domain = domain(dt, model, file=file)

    if domain_name != None and domain_name in dir(data_domain):
      print(f"\n####### Setting up domain: {domain_name} ##########")
      domain_name = domain_name.strip()
      if re.search("\(\)$", domain_name):
        func = f"data_domain.{domain_name}"
      else:
        func = f"data_domain.{domain_name}()"
      eval(func)
    else:
      print(f"No domain found with that name; {domain_name}")
  else:
    data_domain=None
  return data_domain


def IWC_LWC(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True,m_level = [0, 64]):

    for dt in datetime:  # modelrun at time..
        date = dt[0:-2]
        hour = int(dt[-2:])

        param = ['mass_fraction_of_cloud_condensed_water_in_air_ml',
               'mass_fraction_of_cloud_ice_in_air_ml',"air_pressure_at_sea_level","surface_geopotential"]
        check_all = check_data(date=dt, model=model, param=param, levtype="ml", m_level=m_level, step=steps)
        print(check_all.file)
        file_all = check_all.file.loc[0]

        data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_all)

        # lonlat = np.array(data_domain.lonlat)
        print(m_level)
        dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                             date=dt, m_level=m_level)
        print("\n######## Retrieving data ############")
        print(f"--------> from: {dmap_meps.url} ")
        dmap_meps.retrieve()
        dmap_meps.air_pressure_at_sea_level /= 100

        #CALCULATE
        dmap_meps.LWC = np.sum(dmap_meps.mass_fraction_of_cloud_condensed_water_in_air_ml[:,:,:,:],axis=1)
        dmap_meps.LWC =dmap_meps.LWC*1000
        dmap_meps.units.LWC = "g/kg"
        dmap_meps.IWC = np.sum(dmap_meps.mass_fraction_of_cloud_ice_in_air_ml[:,:,:,:],axis=1)
        dmap_meps.IWC = dmap_meps.IWC*1000
        dmap_meps.units.IWC = "g/kg"
        del dmap_meps.mass_fraction_of_cloud_condensed_water_in_air_ml
        del dmap_meps.mass_fraction_of_cloud_ice_in_air_ml
        # for more normal unit of g/m^2 read
        # #https://www.nwpsaf.eu/site/download/documentation/rtm/docs_rttov12/rttov_gas_cloud_aerosol_units.pdf
        # https://www.researchgate.net/post/How-to-convert-the-units-of-specific-cloud-liquid-water-from-ERA5-kg-kg-to-kg-m2
        dmap_meps.LWC[np.where(dmap_meps.LWC < 0.1)] = np.nan
        dmap_meps.IWC[np.where(dmap_meps.IWC < 0.01)] = np.nan

        #It is a bug in pcolormesh. supposedly newest is correct, but not older versions. Invalid corner values set to nan
        #https://github.com/matplotlib/basemap/issues/470
        x,y = np.meshgrid(dmap_meps.x, dmap_meps.y)
        #dlon,dlat=  np.meshgrid(dmap_meps.longitude, dmap_meps.latitude)

        nx, ny = x.shape
        mask = (
          (x[:-1, :-1] > 1e20) |
          (x[1:, :-1] > 1e20) |
          (x[:-1, 1:] > 1e20) |
          (x[1:, 1:] > 1e20) |
          (x[:-1, :-1] > 1e20) |
          (x[1:, :-1] > 1e20) |
          (x[:-1, 1:] > 1e20) |
          (x[1:, 1:] > 1e20)
        )

        # plot map
        lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
        lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
        parallels = dmap_meps.standard_parallel_projection_lambert

        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
        crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                         globe=globe)
        make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))

        for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
                                      
            #ax1 = plt.subplot(projection=crs)

            # determine if image should be created for this time step
            stepok=False
            if tim<25:
                stepok=True
            elif (tim<=36) and ((tim % 3) == 0):
                stepok=True
            elif (tim<=66) and ((tim % 6) == 0):
                stepok=True
            if stepok==True:

                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})
                ttt = tim #+ np.min(steps)
                tidx = tim - np.min(steps)

                ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()

                #pcolor as pcolormesh and  this projection is not happy together. If u want faster, try imshow
                #dmap_meps.LWC[np.where( dmap_meps.LWC <= 0.09)] = np.nan
                data =  dmap_meps.LWC[tidx,:nx - 1, :ny - 1].copy()
                data[mask] = np.nan
                levels = MaxNLocator(nbins=15).tick_values(0.1, 4.0)
                cmap = plt.get_cmap('Reds')
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                CC=ax1.pcolormesh(x, y,  data, shading='flat', cmap=cmap, vmin=0.1, vmax=4.0,zorder=2)
                data =  dmap_meps.IWC[tidx,:nx - 1, :ny - 1].copy()
                data[mask] = np.nan
                levels = MaxNLocator(nbins=15).tick_values(0.01, 0.1)
                cmap = plt.get_cmap('Blues')
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                CI= ax1.pcolormesh(x, y, data, shading='flat', cmap=cmap,alpha=0.5, vmin=0.01, vmax=0.1,zorder=3)

                # MSLP
                # MSLP with contour labels every 10 hPa
                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=4, alpha=1.0,
                                  levels=np.arange(960, 1050, 1),
                                  colors='grey', linewidths=0.5)
                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=5, alpha=1.0,
                                  levels=np.arange(960, 1050, 10),
                                  colors='grey', linewidths=1.0, label="MSLP [hPa]")
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)


                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=6,facecolor="none",edgecolor="gray")  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
                if domain_name != model and data_domain !=None: #weird bug.. cuts off when sees no data value
                     ax1.set_extent(data_domain.lonlat)
                ax1.text(0, 1, "{0}_LWC_IWC_{1}_{2}+{3:02d}".format(model,m_level, dt, ttt), ha='left', va='bottom', \
                                       transform=ax1.transAxes, color='black')
                print("filename: "+make_modelrun_folder + "/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "LWP_IWP", dt, ttt))
                if grid:
                     nicegrid(ax=ax1)

                legend = True
                if legend:
                    cbar = nice_vprof_colorbar(CF=CI, ax=ax1, extend="max",label='IWC [g/kg]',x0=0.75,y0=0.95,width=0.26,height=0.05,
                                               format='%.3f', ticks=[0.001, np.nanmax(dmap_meps.IWC[tidx, :, :])*0.8])
                    cbar = nice_vprof_colorbar(CF=CC, ax=ax1, extend="max", label='LWC [g/kg]',x0=0.50,y0=0.95,width=0.26,height=0.05,
                                              format='%.1f',ticks=[0.09, np.nanmax(dmap_meps.LWC[tidx, :, :])*0.8])
                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                    # proxy.extend(proxy1)
                    lg = ax1.legend(proxy, ["MSLP [hPa]"])
                    frame = lg.get_frame()
                    frame.set_facecolor('white')
                    frame.set_alpha(0.8)

                fig1.savefig(make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "clouds", dt, ttt), bbox_inches="tight", dpi=200)
                ax1.cla()
                plt.clf()
                plt.close(fig1)
    plt.close("all")

if __name__ == "__main__":
  import argparse
  def none_or_str(value):
    if value == 'None':
      return None
    return value
  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--m_level", default=[0,64], nargs="+", type=int, help="suming over modellevels --m_level 30 64 gives loewst 35 modellevels")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()

  #for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
  for s in np.arange(np.min(args.steps), np.max(args.steps)+1, 1):
    IWC_LWC(datetime=args.datetime, steps = [s, s], model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info,grid=args.grid, m_level=args.m_level)

# fin
