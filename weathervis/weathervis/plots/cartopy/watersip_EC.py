# %%
# python watersip_EC.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
#
from weathervis.config import *
from weathervis.utils import *

# import weathervis.config as wc
import cartopy.crs as ccrs
from weathervis.domain import *  # require netcdf4
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import warnings
import cartopy.feature as cfeature
import netCDF4 as nc
import matplotlib.colors as colors

print("done")
# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=Downloading)


def watersip_EC(
    datetime,
    steps=0,
    model="MEPS",
    domain_name=None,
    domain_lonlat=None,
    legend=False,
    info=False,
    save=True,
    grid=True,
    release_name="AN",
):
    # avoid problems when plotting beyond AROME forecast period
    rsteps = steps.copy()
    rsteps[0] = steps[0]
    rsteps[1] = np.min([steps[1], 66])
    for dt in datetime:  # modelrun at time..
        print(dt)
        date = dt[0:-2]
        hour = int(dt[-2:])
        param_sfc = ["air_pressure_at_sea_level", "surface_geopotential"]
        p_level = [850]
        dmet, data_domain, bad_param = checkget_data_handler(
            all_param=param_sfc,
            date=dt,
            model=model,
            step=rsteps,
            p_level=p_level,
            domain_name=domain_name,
            use_latest=False,
        )
        # convert fields
        dmet.air_pressure_at_sea_level /= 100

        # read netcdf files with watersip output
        # dt=2021031600
        cdf = nc.Dataset(
            "/home/centos/watersip/{0}/fc_{1}_{0}_grid_steps.nc".format(
                release_name, dt[0:8]
            ),
            "r",
        )
        lats = cdf.variables["global_latitude"][:]
        lons = cdf.variables["global_longitude"][:]
        # lons, lats = np.meshgrid(lons, lats)
        # print(lons)
        tim = cdf.variables["time"][:]
        ubl = cdf.variables["moisture_uptakes_boundary_layer"][:]
        uft = cdf.variables["moisture_uptakes_free_troposphere"][:]
        upt = ubl + uft

        print(upt.shape)

        # plot map
        lonlat = [
            dmet.longitude[0, 0],
            dmet.longitude[-1, -1],
            dmet.latitude[0, 0],
            dmet.latitude[-1, -1],
        ]
        # print(lonlat)

        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert

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
        # fig1 = plt.figure(figsize=(7, 9))
        make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))

        print(steps)
        for tim in np.arange(np.min(steps), np.max(steps) + 1, 1):
            # ax1 = plt.subplot(projection=crs)

            fig1, ax1 = plt.subplots(
                1, 1, figsize=(7, 9), subplot_kw={"projection": crs}
            )
            tidx = tim - np.min(steps)

            upt_plt = (upt[tim, :, :]).squeeze()

            # print(tidx)
            # print(np.min(upt_plt))
            print(np.max(upt_plt))
            upt_plt = np.where(upt_plt > 1e-20, upt_plt, np.NaN)

            print("Plotting WaterSip-EC {0} + {1:02d} UTC".format(dt, tim * 3))
            # gather, filter and squeeze variables for plotting
            plev = 0
            # reduces noise over mountains by removing values over a certain height.
            F_P = ax1.pcolormesh(
                lons,
                lats,
                upt_plt,
                norm=colors.LogNorm(vmin=1e-20, vmax=1e-11),
                cmap=plt.cm.Blues,
                zorder=1,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
            )
            del upt_plt

            if tim < 66:
                Z = dmet.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(
                    Z < 50000, dmet.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN
                ).squeeze()

                # MSLP with contour labels every 10 hPa
                C_P = ax1.contour(
                    dmet.x,
                    dmet.y,
                    MSLP,
                    zorder=6,
                    alpha=1.0,
                    levels=np.arange(960, 1050, 1),
                    colors="grey",
                    linewidths=0.5,
                    transform=crs,
                )
                C_P = ax1.contour(
                    dmet.x,
                    dmet.y,
                    MSLP,
                    zorder=7,
                    alpha=1.0,
                    levels=np.arange(960, 1050, 10),
                    colors="grey",
                    linewidths=1.0,
                    label="MSLP [hPa]",
                    transform=crs,
                )
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

            ax1.add_feature(cfeature.GSHHSFeature(scale="intermediate"))
            ax1.text(
                0,
                1,
                "{0}_FP_{1}+{2:02d}".format(model, dt, tim * 3),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="black",
            )

            legend = False
            if legend:
                proxy = [
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        fc=pc.get_facecolor()[0],
                    )
                    for pc in CF_T.collections
                ]
                proxy1 = [
                    plt.axhline(y=0, xmin=1, xmax=1, color="red"),
                    plt.axhline(y=0, xmin=1, xmax=1, color="red", linestyle="dashed"),
                    plt.axhline(y=0, xmin=1, xmax=1, color="gray"),
                ]
                proxy.extend(proxy1)
                lg = ax1.legend(
                    proxy,
                    [
                        f"RH > 80% [%] at {dmet.pressure[plev]:.0f} hPa",
                        f"T>0 [C] at {dmet.pressure[plev]:.0f} hPa",
                        f"T<0 [C] at {dmet.pressure[plev]:.0f} hPa",
                        "MSLP [hPa]",
                        "",
                    ],
                )
                frame = lg.get_frame()
                frame.set_facecolor("white")
                frame.set_alpha(1)

            # if info:
            #  plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
            #           fontsize=7)#, bbox=dict(facecolor='white', alpha=0.5))

            if grid:
                nicegrid(ax=ax1)

            # if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
            ax1.set_extent(lonlat)

            model = "WATERSIP_EC"
            print(
                make_modelrun_folder
                + "/{0}_{1}_{2}+{3:02d}.png".format(model, domain_name, dt, tim * 3)
            )
            fig1.savefig(
                make_modelrun_folder
                + "/{0}_{1}_{2}+{3:02d}.png".format(model, domain_name, dt, tim * 3),
                bbox_inches="tight",
                dpi=200,
            )

            ax1.cla()
            plt.clf()
            plt.close(fig1)

    plt.close("all")


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
    parser.add_argument("--release_name", default="AN", help="AN or other sites")
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]"
    )
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--grid", default=True, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()
    print(args.__dict__)

    # split up in 3 retrievals of up to 24h
    watersip_EC(
        datetime=args.datetime,
        steps=[0, np.min([24, np.max(args.steps)])],
        model=args.model,
        domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat,
        legend=args.legend,
        info=args.info,
        grid=args.grid,
        release_name=args.release_name,
    )
    if np.max(args.steps) > 24:
        watersip_EC(
            datetime=args.datetime,
            steps=[27, np.min([36, np.max(args.steps)])],
            model=args.model,
            domain_name=args.domain_name,
            domain_lonlat=args.domain_lonlat,
            legend=args.legend,
            info=args.info,
            grid=args.grid,
            release_name=args.release_name,
        )
    if np.max(args.steps) > 36:
        watersip_EC(
            datetime=args.datetime,
            steps=[42, np.max(args.steps)],
            model=args.model,
            domain_name=args.domain_name,
            domain_lonlat=args.domain_lonlat,
            legend=args.legend,
            info=args.info,
            grid=args.grid,
            release_name=args.release_name,
        )

# fin
