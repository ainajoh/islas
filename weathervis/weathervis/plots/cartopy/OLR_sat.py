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
import pandas as pd
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

def OLR_sat(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True):

  for dt in datetime: #modelrun at time..
    param = ["toa_outgoing_longwave_flux","air_pressure_at_sea_level","surface_geopotential"]
    check_all = check_data(date=dt, model=model, param=param, step=steps)
    file_all = check_all.file
    data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)
    dmap_meps = get_data(model=model, param=param, file=file_all, step=steps, date=dt, data_domain = data_domain)
    dmap_meps.retrieve()

    dmap_meps.air_pressure_at_sea_level/=100


    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    #fig = plt.figure(figsize=(7, 9))
    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                globe=globe)

    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
      fig, ax = plt.subplots(1, 1, figsize=(7, 9),
                               subplot_kw={'projection': crs})

      ttt = tim
      tidx = tim - np.min(steps)
      ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
      MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()

      #ax = plt.subplot(projection=crs)

      print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))
      #ax.coastlines('10m')
      #ax.pcolormesh(dmap_meps.x, dmap_meps.y, dmap_meps.integral_of_toa_outgoing_longwave_flux_wrt_time[0, 0, :, :], vmin=-230,
      #              vmax=-110, cmap=plt.cm.Greys_r)
      #ax.pcolormesh(dmap_meps.x, dmap_meps.y, dmap_meps.toa_outgoing_longwave_flux[tidx, 0, :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)

      # MSLP
      # MSLP with contour labels every 10 hPa
      C_P = ax.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=0.6,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1),
                        colors='cyan', linewidths=0.5)
      C_P = ax.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=0.6,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 5),
                        colors='cyan', linewidths=1.0, label="MSLP [hPa]")
      ax.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

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
      data =  dmap_meps.toa_outgoing_longwave_flux[tidx, 0,:nx - 1, :ny - 1].copy()
      data[mask] = np.nan
      #ax.pcolormesh(x, y, data[ :, :])#, cmap=plt.cm.Greys_r)

      ax.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)
      #ax.pcolormesh(dmap_meps.x, dmap_meps.y, dmap_meps.toa_outgoing_longwave_flux[tidx, 0, :, :], cmap=plt.cm.Greys_r)
      #lat_p = 78.9243
      #lon_p = 11.9312
      #mainpoint = ax.scatter(lon_p, lat_p, s=9.0 ** 2, transform=ccrs.PlateCarree(),
      #                        color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)

      ax.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="brown", linewidth=0.5)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).


      #distancerange="../../data/Table_circle_nm_Andenes.csv"
      #dist = pd.read_csv(distancerange)
      #lats = dist["lat_300nm"]
      #lons = dist["lon_300nm"]
      #lons[dmap_meps.longitude]=np.nan


      #lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
      #lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
      #C300 = ax.plot(lons_mask,lats_mask, transform = ccrs.PlateCarree())

      #lats = dist["lat_400nm"]
      #lons = dist["lon_400nm"]
      #lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
      #lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
      #C300 = ax.plot(lons_mask, lats_mask, transform=ccrs.PlateCarree())

      #lats = dist["lat_500nm"]
      #lons = dist["lon_500nm"]
      #lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
      #lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
      #C300 = ax.plot(lons, lats, transform=ccrs.PlateCarree())

      #lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[0, -1], dmap_meps.latitude[0, 0],
      #          dmap_meps.latitude[-1, -1]]
      #lonlat = [np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude), np.nanmin(dmap_meps.latitude),
      #         np.nanmax(dmap_meps.latitude)]
      #print(dmap_meps.longitude[-2, -2])
      #print(np.nanmax(dmap_meps.longitude))
      #ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
      #ax.set_extent([x[0,0], x[-1,-1], y[0,0], y[-1,-1]], projection=crs)  # (x0, x1, y0, y1)
      #ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
      ax.text(0, 1, "{0}_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', \
               transform=ax.transAxes, color='dimgrey')
      #ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
      #ax.set_extent([lonlat[0]+10, lonlat[1], lonlat[2]-2, lonlat[3]])  # (x0, x1, y0, y1)

      #ax.set_extent((-18.0,80.0,62.0,88.0))  # (x0, x1, y0, y1)

      #ax.set_extent(data_domain.lonlat)
      if grid:
        nicegrid(ax=ax,color="orange")
      fig.savefig(make_modelrun_folder + "/{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(model, domain_name, dt, ttt), bbox_inches="tight", dpi=200)

      ax.cla()
      fig.clf()
      plt.close(fig)

    ax.cla()
    plt.clf()
  plt.close("all")


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
  parser.add_argument("--grid", default=True, help="Display legend")

  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()
  OLR_sat(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info,grid=args.grid)
  #datetime, step=4, model= "MEPS", domain = None
