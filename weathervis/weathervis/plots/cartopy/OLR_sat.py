# %%
# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from add_overlays import *

from weathervis.calculation import *
from weathervis.check_data import *
from weathervis.config import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.utils import *


def OLR_sat(
    datetime,
    steps=0,
    model="MEPS",
    domain_name=None,
    domain_lonlat=None,
    legend=False,
    info=False,
    grid=True,
    track=False,
    runid=None,
    outpath=None,
):
    global OUTPUTPATH
    if outpath != None:
        OUTPUTPATH = outpath

    for dt in datetime:  # modelrun at time..
        if runid != None:
            make_modelrun_folder = setup_directory(
                OUTPUTPATH, "{0}-{1}".format(dt, runid)
            )
        else:
            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))

        param = [
            "toa_outgoing_longwave_flux",
            "air_pressure_at_sea_level",
            "surface_geopotential",
        ]
        check_all = check_data(date=dt, model=model, param=param, step=steps)
        file_all = check_all.file
        data_domain = domain_input_handler(
            dt, model, domain_name, domain_lonlat, file_all
        )
        dmap_meps = get_data(
            model=model,
            param=param,
            file=file_all,
            step=steps,
            date=dt,
            data_domain=data_domain,
        )
        dmap_meps.retrieve()

        # dmap_meps = checkget_data_handler(domain_name=domain_name, model=model, all_param=param, step=steps, date=dt)
        dmap_meps.air_pressure_at_sea_level /= 100

        lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
        lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
        parallels = dmap_meps.standard_parallel_projection_lambert

        # setting up projection
        globe = ccrs.Globe(
            ellipse="sphere", semimajor_axis=6371000.0, semiminor_axis=6371000.0
        )
        crs = ccrs.LambertConformal(
            central_longitude=lon0,
            central_latitude=lat0,
            standard_parallels=parallels,
            globe=globe,
        )

        for tim in np.arange(np.min(steps), np.max(steps) + 1, 1):
            fig, ax = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})

            # determine if image should be created for this time step
            stepok = False
            if tim < 25:
                stepok = True
            elif (tim <= 36) and ((tim % 3) == 0):
                stepok = True
            elif (tim <= 66) and ((tim % 6) == 0):
                stepok = True
            if stepok == True:

                ttt = tim
                tidx = tim - np.min(steps)
                ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(
                    ZS < 3000,
                    dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :],
                    np.NaN,
                ).squeeze()
                if track:
                    gca = plt.gca()
                    tt = dmap_meps.time[tidx]
                    sc1 = plot_track_on_map(
                        dt, model, tim, gca, ccrs, "#FFFFFF", "#FF0000", tt
                    )

                # MSLP
                # MSLP with contour labels every 10 hPa
                C_P = ax.contour(
                    dmap_meps.x,
                    dmap_meps.y,
                    MSLP,
                    zorder=10,
                    alpha=0.6,
                    levels=np.arange(
                        round(np.nanmin(MSLP), -1) - 10,
                        round(np.nanmax(MSLP), -1) + 10,
                        1,
                    ),
                    colors="cyan",
                    linewidths=0.5,
                )
                C_P = ax.contour(
                    dmap_meps.x,
                    dmap_meps.y,
                    MSLP,
                    zorder=10,
                    alpha=0.6,
                    levels=np.arange(
                        round(np.nanmin(MSLP), -1) - 10,
                        round(np.nanmax(MSLP), -1) + 10,
                        5,
                    ),
                    colors="cyan",
                    linewidths=1.0,
                    label="MSLP [hPa]",
                )
                ax.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                # It is a bug in pcolormesh. supposedly newest is correct, but not older versions. Invalid corner values set to nan
                # https://github.com/matplotlib/basemap/issues/470
                x, y = np.meshgrid(dmap_meps.x, dmap_meps.y)

                nx, ny = x.shape
                mask = (
                    (x[:-1, :-1] > 1e20)
                    | (x[1:, :-1] > 1e20)
                    | (x[:-1, 1:] > 1e20)
                    | (x[1:, 1:] > 1e20)
                    | (x[:-1, :-1] > 1e20)
                    | (x[1:, :-1] > 1e20)
                    | (x[:-1, 1:] > 1e20)
                    | (x[1:, 1:] > 1e20)
                )
                data = dmap_meps.toa_outgoing_longwave_flux[
                    tidx, 0, : nx - 1, : ny - 1
                ].copy()
                data[mask] = np.nan

                ax.pcolormesh(
                    x, y, data[:, :], vmin=-230, vmax=-110, cmap=plt.cm.Greys_r
                )
                # lat_p = 78.9243
                # lon_p = 11.9312
                # mainpoint = ax.scatter(lon_p, lat_p, s=9.0 ** 2, transform=ccrs.PlateCarree(),
                #                        color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)

                ax.add_feature(
                    cfeature.GSHHSFeature(scale="intermediate"),
                    edgecolor="brown",
                    linewidth=0.5,
                )

                # distancerange="../../data/Table_circle_nm_Andenes.csv"
                # dist = pd.read_csv(distancerange)
                # lats = dist["lat_300nm"]
                # lons = dist["lon_300nm"]
                # lons[dmap_meps.longitude]=np.nan

                # lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
                # lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
                # C300 = ax.plot(lons_mask,lats_mask, transform = ccrs.PlateCarree())

                # lats = dist["lat_400nm"]
                # lons = dist["lon_400nm"]
                # lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
                # lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
                # C300 = ax.plot(lons_mask, lats_mask, transform=ccrs.PlateCarree())

                # lats = dist["lat_500nm"]
                # lons = dist["lon_500nm"]
                # lons_mask = ma.masked_outside(lons, np.nanmin(dmap_meps.longitude), np.nanmax(dmap_meps.longitude))
                # lats_mask = ma.masked_outside(lats, np.nanmin(dmap_meps.latitude), np.nanmax(dmap_meps.latitude))
                # C300 = ax.plot(lons, lats, transform=ccrs.PlateCarree())

                # ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
                ax.text(
                    0,
                    1,
                    "{0}_{1}+{2:02d}".format(model, dt, ttt),
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    color="dimgrey",
                )
                if grid:
                    nicegrid(ax=ax, color="orange")

                add_ISLAS_overlays(ax)

                filename = "{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(
                    model, domain_name, dt, ttt
                )

                print("Plotting {0} + {1:02d} UTC".format(dt, ttt))
                print(
                    make_modelrun_folder
                    + "/{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(
                        model, domain_name, dt, ttt
                    )
                )
                fig.savefig(
                    make_modelrun_folder
                    + "/{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(
                        model, domain_name, dt, ttt
                    ),
                    bbox_inches="tight",
                    dpi=200,
                )

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
        if value == "None":
            return None
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+"
    )
    parser.add_argument(
        "--steps",
        default=0,
        nargs="+",
        type=int,
        help="forecast times example --steps 0 3 gives time 0 to 3",
    )
    parser.add_argument("--model", default="MEPS", help="MEPS or AromeArctic")
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]"
    )
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--grid", default=True, help="Display legend")
    parser.add_argument("--track", default=False, help="Display legend", type=bool)
    parser.add_argument("--id", default=None, help="Display legend", type=str)
    parser.add_argument("--outpath", default=None, help="Display legend", type=str)

    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    OLR_sat(
        datetime=args.datetime,
        steps=args.steps,
        model=args.model,
        domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat,
        legend=args.legend,
        info=args.info,
        grid=args.grid,
        track=args.track,
        runid=args.id,
        outpath=args.outpath,
    )
    # datetime, step=4, model= "MEPS", domain = None
