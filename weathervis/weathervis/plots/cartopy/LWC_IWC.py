# %%
# python LWC_IWC.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from add_overlays import *

from weathervis.calculation import *
from weathervis.check_data import *
from weathervis.config import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.utils import *

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)


def IWC_LWC(
    datetime,
    steps=0,
    model="MEPS",
    domain_name=None,
    domain_lonlat=None,
    legend=False,
    info=False,
    grid=True,
    m_level=[0, 64],
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

    for dt in datetime:  # modelrun at time..
        date = dt[0:-2]
        hour = int(dt[-2:])

        param = [
            "mass_fraction_of_cloud_condensed_water_in_air_ml",
            "surface_air_pressure",
            "mass_fraction_of_cloud_ice_in_air_ml",
            "air_pressure_at_sea_level",
            "surface_geopotential",
        ]
        check_all = check_data(
            date=dt, model=model, param=param, levtype="ml", m_level=m_level, step=steps
        )
        # print(check_all.file)
        file_all = check_all.file.loc[0]

        data_domain = domain_input_handler(
            dt, model, domain_name, domain_lonlat, file_all
        )

        # lonlat = np.array(data_domain.lonlat)
        # print(m_level)
        dmet = get_data(
            model=model,
            data_domain=data_domain,
            param=param,
            file=file_all,
            step=steps,
            date=dt,
            m_level=m_level,
        )
        # print("\n######## Retrieving data ############")
        # print(f"--------> from: {dmet.url} ")
        dmet.retrieve()
        dmet.air_pressure_at_sea_level /= 100

        # It is a bug in pcolormesh. supposedly newest is correct, but not older versions. Invalid corner values set to nan
        # https://github.com/matplotlib/basemap/issues/470
        x, y = np.meshgrid(dmet.x, dmet.y)
        # dlon,dlat=  np.meshgrid(dmet.longitude, dmet.latitude)

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

        # plot map
        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert

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

            # ax1 = plt.subplot(projection=crs)

            # determine if image should be created for this time step
            stepok = False
            if tim < 25:
                stepok = True
            elif (tim <= 36) and ((tim % 3) == 0):
                stepok = True
            elif (tim <= 66) and ((tim % 6) == 0):
                stepok = True
            if stepok == True:

                tidx = tim - np.min(steps)

                # calculate LWC
                LWC = dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[
                    tidx, :, :, :
                ]
                # p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
                dp = np.insert(
                    np.diff(
                        np.mean(dmet.surface_air_pressure[tidx, 0, :, :].squeeze())
                        * dmet.b
                        + dmet.ap
                    ),
                    0,
                    0,
                )
                shape = np.swapaxes(LWC, LWC.ndim - 1, 0).shape
                B_brc = np.broadcast_to(dp, shape)
                # Swap back the axes. As before, this only changes our "point of view".
                B_brc = np.swapaxes(B_brc, LWC.ndim - 1, 0)
                LWC = np.sum(LWC * B_brc, axis=0) / 9.81

                IWC = dmet.mass_fraction_of_cloud_ice_in_air_ml[tidx, :, :, :]
                shape = np.swapaxes(IWC, IWC.ndim - 1, 0).shape
                B_brc = np.broadcast_to(dp, shape)
                # Swap back the axes. As before, this only changes our "point of view".
                B_brc = np.swapaxes(B_brc, IWC.ndim - 1, 0)
                IWC = np.sum(IWC * B_brc, axis=0) / 9.81

                # dmet.LWC = np.sum(dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[:,:,:,:],axis=1)
                # dmet.LWC =dmet.LWC*1000
                dmet.units.LWC = "mm"
                # dmet.IWC = np.sum(dmet.mass_fraction_of_cloud_ice_in_air_ml[:,:,:,:],axis=1)
                # dmet.IWC = dmet.IWC*1000
                dmet.units.IWC = "mm"

                # for more normal unit of g/m^2 read
                # #https://www.nwpsaf.eu/site/download/documentation/rtm/docs_rttov12/rttov_gas_cloud_aerosol_units.pdf
                # https://www.researchgate.net/post/How-to-convert-the-units-of-specific-cloud-liquid-water-from-ERA5-kg-kg-to-kg-m2
                # print('LWC')
                # print(np.max(LWC))
                # print('IWC')
                # print(np.max(IWC))
                LWC[np.where(LWC < 0.002)] = np.nan
                IWC[np.where(IWC < 0.0005)] = np.nan

                fig1, ax1 = plt.subplots(
                    1, 1, figsize=(7, 9), subplot_kw={"projection": crs}
                )

                ZS = dmet.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(
                    ZS < 3000, dmet.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN
                ).squeeze()

                # pcolor as pcolormesh and  this projection is not happy together. If u want faster, try imshow
                # dmet.LWC[np.where( dmet.LWC <= 0.09)] = np.nan
                ldata = LWC[: nx - 1, : ny - 1].copy()
                ldata[mask] = np.nan
                idata = IWC[: nx - 1, : ny - 1].copy()
                idata[mask] = np.nan

                CI = ax1.pcolormesh(
                    x,
                    y,
                    idata[:, :],
                    cmap=plt.cm.Blues,
                    alpha=1,
                    vmin=0.0005,
                    vmax=0.04,
                    zorder=1,
                )
                CC = ax1.pcolormesh(
                    x, y, ldata[:, :], cmap=plt.cm.Reds, vmin=0.002, vmax=0.1, zorder=2
                )
                IWC[np.where(IWC > 0.005)] = np.nan
                CI = ax1.pcolormesh(
                    x,
                    y,
                    idata[:, :],
                    cmap=plt.cm.Blues,
                    alpha=0.3,
                    vmin=0.0005,
                    vmax=0.04,
                    zorder=3,
                )

                # MSLP
                # MSLP with contour labels every 10 hPa
                C_P = ax1.contour(
                    dmet.x,
                    dmet.y,
                    MSLP,
                    zorder=4,
                    alpha=1.0,
                    levels=np.arange(960, 1050, 1),
                    colors="grey",
                    linewidths=0.5,
                )
                C_P = ax1.contour(
                    dmet.x,
                    dmet.y,
                    MSLP,
                    zorder=5,
                    alpha=1.0,
                    levels=np.arange(960, 1050, 10),
                    colors="grey",
                    linewidths=1.0,
                    label="MSLP [hPa]",
                )
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                ax1.add_feature(
                    cfeature.GSHHSFeature(scale="intermediate"),
                    zorder=6,
                    facecolor="none",
                    edgecolor="gray",
                )
                if (
                    domain_name != model and data_domain != None
                ):  # weird bug.. cuts off when sees no data value
                    ax1.set_extent(data_domain.lonlat)
                ax1.text(
                    0,
                    1,
                    "{0}_LWC_IWC_{1}_{2}+{3:02d}".format(model, m_level, dt, tim),
                    ha="left",
                    va="bottom",
                    transform=ax1.transAxes,
                    color="black",
                )
                if grid:
                    nicegrid(ax=ax1)

                add_ISLAS_overlays(ax1)

                legend = True
                if legend:
                    cbar = nice_vprof_colorbar(
                        CF=CI,
                        ax=ax1,
                        extend="max",
                        label="IWC [g/kg]",
                        x0=0.75,
                        y0=0.95,
                        width=0.26,
                        height=0.05,
                        format="%.3f",
                        ticks=[0.001, np.nanmax(IWC[:, :]) * 0.8],
                    )
                    cbar = nice_vprof_colorbar(
                        CF=CC,
                        ax=ax1,
                        extend="max",
                        label="LWC [g/kg]",
                        x0=0.50,
                        y0=0.95,
                        width=0.26,
                        height=0.05,
                        format="%.1f",
                        ticks=[0.09, np.nanmax(LWC[:, :]) * 0.8],
                    )
                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray", zorder=7)]
                    # proxy.extend(proxy1)
                    # legend's location fixed, otherwise it takes very long to find optimal spot
                    lg = ax1.legend(proxy, ["MSLP [hPa]"], loc="upper left")
                    frame = lg.get_frame()
                    frame.set_facecolor("white")
                    frame.set_alpha(0.8)

                # print('Plotting {0} + {1:02d} UTC'.format(dt, tim))
                # print(make_modelrun_folder + "/{0}_{1}_clouds_{2}+{3:02d}.png".format(model, domain_name, dt, tim))
                print(
                    "filename: "
                    + make_modelrun_folder
                    + "/{0}_{1}_{2}_{3}+{4:02d}.png".format(
                        model, domain_name, "clouds", dt, tim
                    )
                )
                fig1.savefig(
                    make_modelrun_folder
                    + "/{0}_{1}_clouds_{2}+{3:02d}.png".format(
                        model, domain_name, dt, tim
                    ),
                    bbox_inches="tight",
                    dpi=200,
                )
                ax1.cla()
                plt.clf()
                plt.close(fig1)
        # ax1.cla()
        # plt.clf()
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
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]"
    )
    parser.add_argument(
        "--m_level",
        default=[0, 64],
        nargs="+",
        type=int,
        help="suming over modellevels --m_level 30 64 gives loewst 35 modellevels",
    )
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--grid", default=True, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    parser.add_argument("--id", default=None, help="Display legend", type=str)
    parser.add_argument("--outpath", default=None, help="Display legend", type=str)
    args = parser.parse_args()

    # CHUNCK SIZE TO BIG
    s = np.arange(np.min(args.steps), np.max(args.steps) + 1)
    cn = np.int(len(s) // 6)
    if cn == 0:  # length of 6 not exceeded
        IWC_LWC(
            datetime=args.datetime,
            steps=[np.min(args.steps), np.max(args.steps)],
            model=args.model,
            domain_name=args.domain_name,
            domain_lonlat=args.domain_lonlat,
            legend=args.legend,
            info=args.info,
            grid=args.grid,
            m_level=args.m_level,
            runid=args.id,
            outpath=args.outpath,
        )
    else:  # lenght of 6 is exceeded, split in chunks, set by cn+1
        print(
            f"\n####### request exceeds 6 timesteps, will be chunked to smaller bits due to request limit ##########"
        )
        chunks = np.array_split(s, cn + 1)
        for c in chunks:
            IWC_LWC(
                datetime=args.datetime,
                steps=[np.min(c), np.max(c)],
                model=args.model,
                domain_name=args.domain_name,
                domain_lonlat=args.domain_lonlat,
                legend=args.legend,
                info=args.info,
                grid=args.grid,
                m_level=args.m_level,
                runid=args.id,
                outpath=args.outpath,
            )
# fin
