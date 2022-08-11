# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from weathervis.calculation import *
from weathervis.check_data import *
from weathervis.config import *
from weathervis.domain import *  # require netcdf4
from weathervis.get_data import *
from weathervis.utils import domain_input_handler

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)


def Z500_VEL(
    datetime,
    steps=0,
    model="MEPS",
    domain_name=None,
    domain_lonlat=None,
    legend=False,
    info=False,
):

    for dt in datetime:  # modelrun at time..
        date = dt[0:-2]
        hour = int(dt[-2:])
        param_sfc = [
            "air_pressure_at_sea_level",
            "precipitation_amount_acc",
            "surface_geopotential",
        ]
        param_pl = ["x_wind_pl", "y_wind_pl", "geopotential_pl"]
        param = param_sfc + param_pl
        split = False
        print("\n######## Checking if your request is possibel ############")
        try:
            check_all = check_data(date=dt, model=model, param=param, p_level=500)
        except ValueError:
            split = True
            try:
                print("--------> Splitting up your request to find match ############")
                check_sfc = check_data(date=dt, model=model, param=param_sfc)
                check_pl = check_data(date=dt, model=model, param=param_pl, p_level=500)
            except ValueError:
                print(
                    "!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!"
                )
                break
        print("--------> Found match for your request ############")

        if not split:
            file_all = check_all.file.loc[0]

            data_domain = domain_input_handler(
                dt,
                model,
                file_all,
                domain_name=domain_name,
                domain_lonlat=domain_lonlat,
            )

            # lonlat = np.array(data_domain.lonlat)
            dmap_meps = get_data(
                model=model,
                data_domain=data_domain,
                param=param,
                file=file_all,
                step=steps,
                date=dt,
                p_level=500,
            )
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmap_meps.url} ")
            dmap_meps.retrieve()
            tmap_meps = dmap_meps  # two names for same value, no copying done.
        else:
            # get sfc level data
            file_sfc = check_sfc.file.loc[0]
            data_domain = domain_input_handler(
                dt,
                model,
                file_sfc,
                domain_name=domain_name,
                domain_lonlat=domain_lonlat,
            )
            # lonlat = np.array(data_domain.lonlat)
            dmap_meps = get_data(
                model=model,
                param=param_sfc,
                file=file_sfc,
                step=steps,
                date=dt,
                data_domain=data_domain,
            )
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmap_meps.url} ")
            dmap_meps.retrieve()

            # get pressure level data
            file_pl = check_pl.file
            tmap_meps = get_data(
                model=model,
                data_domain=data_domain,
                param=param_pl,
                file=file_pl,
                step=steps,
                date=dt,
                p_level=500,
            )
            print("\n######## Retriving data ############")
            print(f"--------> from: {tmap_meps.url} ")
            tmap_meps.retrieve()

        print("DONE RETRIEVE")
        # convert fields
        print("cond airpressure")
        dmap_meps.air_pressure_at_sea_level /= 100
        print("cond geopotential_pl")
        tmap_meps.geopotential_pl /= 10.0
        print("cond uv")
        u, v = xwind2uwind(tmap_meps.x_wind_pl, tmap_meps.y_wind_pl, tmap_meps.alpha)
        print("cond vel")
        vel = wind_speed(tmap_meps.x_wind_pl, tmap_meps.y_wind_pl)

        print("map setup")

        # plot map
        fig1, ax1 = plt.subplots(figsize=(7, 9))

        lonlat = [
            dmap_meps.longitude[0, 0],
            dmap_meps.longitude[-1, -1],
            dmap_meps.latitude[0, 0],
            dmap_meps.latitude[-1, -1],
        ]

        lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
        lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
        print("map =map")
        map = Basemap(
            llcrnrlon=lonlat[0],
            llcrnrlat=lonlat[2],
            urcrnrlon=lonlat[1],
            urcrnrlat=lonlat[3],
            resolution="i",
            projection="lcc",
            lon_0=lon0,
            lat_0=lat0,
            lat_1=lat0,
            area_thresh=0.0001,
        )
        print("map x y")
        x, y = map(dmap_meps.longitude, dmap_meps.latitude)
        print("start for loop")
        for tim in np.arange(np.min(steps), np.max(steps) + 1, 1):
            tidx = tim - np.min(steps)

            print("Plotting {0} + {1:02d} UTC".format(dt, tim))

            # gather, filter and squeeze variables for plotting
            # plev1 = 3
            plev2 = 0
            ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
            MSLP = np.where(
                ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN
            ).squeeze()
            TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=1)[
                tidx, 0, :, :
            ].squeeze()
            VEL = (vel[tidx, plev2, :, :]).squeeze()
            Z = (tmap_meps.geopotential_pl[tidx, plev2, :, :]).squeeze()
            # velocity bars
            # U = (u[tim, plev2, :, :]).squeeze()
            # V = (v[tim, plev2, :, :]).squeeze()
            # Rotation in basemap: I thought it was not needed when plotting in same proj as model.
            # https://psysmon.mertl-research.at/sourcedoc/autogen/psysmon.packages.geometry.editGeometry.Basemap.rotate_vector.html
            # https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
            # Ue, Ve = map.rotate_vector(U, V, tmap_meps.longitude, tmap_meps.latitude) #so that shapes in plot is relative correct to shape in wind?
            # clear subplot
            # plt.cla()
            cmap = plt.get_cmap(
                "viridis"
            )  # PuBuGn PuBuGn, nipy_spectral twilight  , plasma, gist_ncar viridis  inferno ,,,rainbow
            # lvl = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            lvl = [0.01, 0.2, 0.5, 1, 3, 5, 10, 15, 20, 25, 30]
            norm = mcolors.BoundaryNorm(lvl, cmap.N)

            try:  # workaround for a stupid matplotlib error not handling when all values are outside of range in lvl or all just nans..
                # https://github.com/SciTools/cartopy/issues/1290
                CF_prec = ax1.contourf(
                    x,
                    y,
                    TP,
                    zorder=10,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.4,
                    antialiased=True,
                    levels=lvl,
                    extend="max",
                )  #
            except:
                pass
            # MSLP with contour labels every 10 hPa
            C_P = ax1.contour(
                x,
                y,
                MSLP,
                zorder=1,
                alpha=1.0,
                levels=np.arange(
                    round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1
                ),
                colors="grey",
                linewidths=0.5,
            )
            C_P = ax1.contour(
                x,
                y,
                MSLP,
                zorder=2,
                alpha=1.0,
                levels=np.arange(
                    round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 10
                ),
                colors="grey",
                linewidths=1.0,
                label="MSLP [hPa]",
            )
            ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)
            # skip = (slice(None, None, 7), slice(None, None, 7))
            # Cq = plt.barbs(x[skip][skip],y[skip][skip],Ue[skip][skip],Ve[skip][skip], zorder=1000, color = "r")
            CS = ax1.contour(
                x,
                y,
                VEL,
                zorder=3,
                alpha=1.0,
                levels=np.arange(-80, 80, 5),
                colors="green",
                linewidths=0.7,
            )
            # geopotential
            CS = ax1.contour(
                x,
                y,
                Z,
                zorder=3,
                alpha=1.0,
                levels=np.arange(4600, 5800, 20),
                colors="blue",
                linewidths=0.7,
            )
            ax1.clabel(CS, CS.levels, inline=True, fmt="%4.0f", fontsize=10)

            map.drawcoastlines(linewidth=0.5, color="black", ax=ax1, zorder=5)
            ##########################################################

            if legend:
                proxy = [
                    plt.axhline(y=0, xmin=1, xmax=1, color="green"),
                    plt.axhline(y=0, xmin=1, xmax=1, color="blue"),
                ]
                plt.colorbar(CF_prec, fraction=0.046, pad=0.04)
                lg = ax1.legend(
                    proxy,
                    [
                        f"Wind strength [m/s] at {dmap_meps.pressure[plev2]:.0f} hPa",
                        f"Geopotential [{tmap_meps.units_geopotential_pl}] at {dmap_meps.pressure[plev2]:.0f} hPa",
                    ],
                )
                frame = lg.get_frame()
                frame.set_facecolor("white")
                frame.set_alpha(1)

            if info:
                plt.text(
                    x=0,
                    y=-1,
                    s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
                    fontsize=7,
                )  # , bbox=dict(facecolor='white', alpha=0.5))

                # plt.show()
            fig1.savefig(
                "../../../output/{0}_Z500_VEL_P_{1}+{2:02d}.png".format(model, dt, tim),
                bbox_inches="tight",
                dpi=200,
            )
            ax1.cla()

        proxy = [
            plt.axhline(y=0, xmin=1, xmax=1, color="green"),
            plt.axhline(y=0, xmin=1, xmax=1, color="blue"),
        ]

        fig2 = plt.figure(figsize=(2, 1.25))
        fig2, ax2 = plt.subplots()
        fig2.legend(
            proxy,
            [
                f"Wind strength [m/s] at {dmap_meps.pressure[plev2]:.0f} hPa",
                f"Geopotential [{tmap_meps.units_geopotential_pl}]{dmap_meps.pressure[plev2]:.0f} hPa",
            ],
        )
        fig2.savefig(
            "../../../output/{0}_Z500_VEL_P_LEGEND.png".format(model),
            bbox_inches="tight",
            dpi=200,
        )

        fig3, ax3 = plt.subplots()
        fig3.colorbar(CF_prec, fraction=0.046, pad=0.04)
        ax3.remove()
        fig3.savefig(
            "../../../output/{0}_Z500_VEL_P_COLORBAR.png".format(model),
            bbox_inches="tight",
            dpi=200,
        )

    plt.clf()


# fin

if __name__ == "__main__":
    import argparse

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
    parser.add_argument("--domain_name", default=None, help="MEPS or AromeArctic")
    parser.add_argument(
        "--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]"
    )
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()
    Z500_VEL(
        datetime=args.datetime,
        steps=args.steps,
        model=args.model,
        domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat,
        legend=args.legend,
        info=args.info,
    )
    # datetime, step=4, model= "MEPS", domain = None
