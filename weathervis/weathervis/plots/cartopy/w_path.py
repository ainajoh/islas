
# %%
# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

from weathervis.config import *
from weathervis.utils import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def IWP_and_LWP():
    # calculate liquid cloud water path and liquid cloud ice path for lowermost 35 levels,
    var = ['mass_fraction_of_cloud_condensed_water_in_air_ml',
           'mass_fraction_of_cloud_ice_in_air_ml']
    nam = ['LWP_35','IWP_35']
    mod = ['cy40_ref','cy40_LTOT']
    data = [data_full,data_LTOT]
    IWIP = {}
    t = 60
    for m,dat in zip(mod,data):
        IWIP[m]={}
        for s,n in zip(var,nam):
            buf = dat.variables[s][t,:,:,:]
            IWIP[m][n] = np.sum(buf[30:65,:,:],axis=0)

def Z500_VEL(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False):

    for dt in datetime:  # modelrun at time..
        date = dt[0:-2]
        hour = int(dt[-2:])
        m_level = [30, 64]

        param = ['mass_fraction_of_cloud_condensed_water_in_air_ml',
               'mass_fraction_of_cloud_ice_in_air_ml']
        check_all = check_data(date=dt, model=model, param=param, levtype="ml", m_level=m_level, step=steps)
        print(check_all.file)
        file_all = check_all.file.loc[0]

        data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_all)

        # lonlat = np.array(data_domain.lonlat)
        dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                             date=dt, m_level=m_level)
        print("\n######## Retrieving data ############")
        print(f"--------> from: {dmap_meps.url} ")
        dmap_meps.retrieve()

        #CALCULATE
        dmap_meps.LWP_35 = np.sum(dmap_meps.mass_fraction_of_cloud_condensed_water_in_air_ml[:,:,:,:],axis=1)
        dmap_meps.LWP_35 =dmap_meps.LWP_35*1000
        dmap_meps.units.LWP_35 = "g/kg"
        dmap_meps.IWP_35 = np.sum(dmap_meps.mass_fraction_of_cloud_ice_in_air_ml[:,:,:,:],axis=1)
        dmap_meps.IWP_35 = dmap_meps.IWP_35*1000
        dmap_meps.units.IWP_35 = "g/kg"
        # for more normal unit of g/m^2 read
        # #https://www.nwpsaf.eu/site/download/documentation/rtm/docs_rttov12/rttov_gas_cloud_aerosol_units.pdf
        # https://www.researchgate.net/post/How-to-convert-the-units-of-specific-cloud-liquid-water-from-ERA5-kg-kg-to-kg-m2

        # plot map

        lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
        lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
        parallels = dmap_meps.standard_parallel_projection_lambert

        # setting up projection
        # setting up projection
        C = [[205, 218, 254],
             [166, 185, 254],
             [124, 159, 255],
             [108, 126, 255],
             [79, 85, 245],
             [8, 68, 255],
             [16, 135, 132],
             [88, 255, 6],
             [136, 255, 11],
             [188, 255, 40],
             [254, 255, 9],
             [254, 187, 9],
             [254, 142, 8],
             [251, 102, 9],
             [250, 0, 7]]
        C = np.array(C)
        C = np.divide(C, 255.)  # RGB has to be between 0 and 1 in python

        for vvar in ["LWP_35","IWP_35"]:
            fig1 = plt.figure(figsize=(7, 9))
            globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
            crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                         globe=globe)
            for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
                ax1 = plt.subplot(projection=crs)
                ttt = tim #+ np.min(steps)
                tidx = tim - np.min(steps)
                print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))
                if vvar=="LWP_35":
                    lvl = np.linspace(1,30,16)
                else:
                    lvl = np.linspace(0.001,0.1,16)

                CF_prec = ax1.contourf(dmap_meps.x, dmap_meps.y, getattr(dmap_meps, vvar)[tidx, :, :] , levels=lvl, colors=C, zorder=0,
                                antialiased=True,extend = "max")
                plt.colorbar(CF_prec, fraction=0.046, pad=0.01, aspect=25, label=f"{vvar}[{ getattr(dmap_meps.units, vvar)}]",
                             extend="both")
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).

                make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
                print("filename: "+make_modelrun_folder + "/{0}_{1}_{2}_{3}_+{4:02d}.png".format(model, domain_name, vvar, dt, ttt))
                fig1.savefig(make_modelrun_folder +"/{0}_{1}_{2}_{3}_+{4:02d}.png".format(model, domain_name, vvar, dt, ttt), bbox_inches="tight", dpi=200)
                ax1.cla()
                plt.clf()



# fin

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
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()
  Z500_VEL(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info)
  #datetime, step=4, model= "MEPS", domain = None





