import os
import sys
from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import (
    shapereader,  # For reading shapefiles containg high-resolution coastline.
)
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from weathervis.calculation import *
from weathervis.config import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.utils import *

# (camp1) c02z62belvdl:cartopy ainajoh$ python plot_meteogram.py --datetime 2018031700 --point_num 1 --steps 0 60 --model AromeArctic --domain_lonlat 15.8 16.4 69.2 69.4 --point_lonlat 16.120 69.310

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
# package_path = os.path.dirname(__file__)
os.chdir(dname)


# matplotlib.rcParams.update({'figure.autolayout': True})
def domain_input_handler(dt, model, domain_name, domain_lonlat, file):
    if domain_name or domain_lonlat:
        if domain_lonlat:
            print(
                f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########"
            )
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
        data_domain = None
    return data_domain


def nice_vprof_colorbar(
    CF,
    ax,
    lvl=None,
    ticks=None,
    label=None,
    highlight_val=None,
    highlight_linestyle="k--",
    extend="both",
):
    x0, y0, width, height = 0.75, 0.86, 0.26, 0.13
    axins = inset_axes(
        ax,
        width="80%",
        height="23%",
        bbox_to_anchor=(x0, y0, width, height),  # (x0, y0, width, height)
        bbox_transform=ax.transAxes,
        loc="upper center",
    )
    cbar = plt.colorbar(
        CF,
        extend=extend,
        cax=axins,
        orientation="horizontal",
        ticks=ticks,
        format="%.1f",
    )
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            width,
            height,
            fc=[1, 1, 1, 0.7],
            transform=ax.transAxes,
            zorder=1000,
        )
    )

    if ticks is not None:
        cbar.ax.xaxis.set_tick_params(pad=-0.5)  # all ticks closer attatched to bar

    if highlight_val is not None:
        # only for equally spaced lvls now. Maybe use diff later to correct for irregularies
        # diff = [t[i + 1] - t[i] for i in range(len(t) - 1)] #list of diff between element
        len = lvl[-1] - lvl[0]  # 17
        us = highlight_val - lvl[0]
        loc_ticks = us / float(len)  # 0.5625, 0,975862068965517

        cbar.ax.plot(
            [loc_ticks] * 2, [0, 1], highlight_linestyle
        )  # additional contour on plot

    if label is not None:
        cbar.set_label(label, labelpad=-0.5)
    return cbar


def scewT(dmet, dmet_sfx, dmet_ml, jindx, iindx, dirName_b1, figname_b1, ip):
    import os
    from collections import UserDict

    import matplotlib.pyplot as plt
    import metpy.calc as mpcalc
    import metpy.plots as mplt
    import numpy as np
    from metpy.plots import SkewT
    from metpy.units import units
    from netCDF4 import Dataset

    pass


def plot_meteogram_vertical(
    dmet, dmet_sfx, dmet_ml, jindx, iindx, dirName_b1, figname_b1, ip
):

    # UNIT and CALULATIONS:
    p = dmet_ml.p[:, :, jindx, iindx] / 100
    specific_humidity_ml = dmet_ml.specific_humidity_ml * 1000
    ur, vr = xwind2uwind(dmet.x_wind_ml, dmet.y_wind_ml, dmet.alpha)
    vel = wind_speed(dmet.x_wind_ml, dmet.y_wind_ml)
    vel_dir = wind_dir(dmet.x_wind_ml, dmet.y_wind_ml, dmet.alpha)
    air_tempC = dmet.air_temperature_ml - 273.15
    p_top = 500
    z = pl2alt_full2full_gl(
        dmet.air_temperature_ml, dmet.specific_humidity_ml, dmet_ml.p
    )
    # rh_p = relative_humidity(temp_p, q_p, p_p)

    # dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600
    temp_p = np.ma.array(dmet.air_temperature_ml[:, :, jindx, iindx], mask=p < p_top)
    uml_p = np.ma.array(ur[:, :, jindx, iindx], mask=p < p_top)
    vml_p = np.ma.array(vr[:, :, jindx, iindx], mask=p < p_top)
    p_p = np.ma.array(p, mask=p > p_top)
    vel_p = np.ma.array(vel[:, :, jindx, iindx], mask=p < p_top)
    q_p = np.ma.array(specific_humidity_ml[:, :, jindx, iindx], mask=p < p_top)
    dtdz_p = np.ma.array(dmet.dtdz[:, :, jindx, iindx], mask=p < p_top)
    air_tempC_p = np.ma.array(air_tempC[:, :, jindx, iindx], mask=p < p_top)
    # massfrac_cloud = np.ma.array(dmet_ml.mass_fraction_of_cloud_condensed_water_in_air_ml[:, :, jindx, iindx], mask=p < p_top)
    areafrac_cloud = np.ma.array(
        dmet_ml.cloud_area_fraction_ml[:, :, jindx, iindx], mask=p < p_top
    )
    areafrac_cloud = areafrac_cloud * 100
    rh_p = relative_humidity(
        air_tempC_p,
        dmet_ml.specific_humidity_ml[:, :, jindx, iindx],
        dmet_ml.p[:, :, jindx, iindx],
    )
    # Ri_p = np.ma.array(dmet.Ri[:, :, jindx, iindx], mask=p < p_top)
    # c_base_pp = dmet_ml.cloud_base_altitude[:, :, jindx, iindx]
    # c_top_pp = dmet_ml.cloud_top_altitude[:, :, jindx, iindx]
    # c_base__alt_p = alt_sl2pl(dmet.surface_air_pressure, c_base_pp)
    # point_alt_sl2pres_old(jindx, iindx, c_base_pp, data_altitude_sl, t_v_level, p, surface_air_pressure,
    #                      surface_geopotential

    z_p = np.ma.array(z[:, :, jindx, iindx], mask=p < p_top)

    # mass_fraction_of_graupel_in_air_ml
    # mass_fraction_of_rain_in_air_ml
    # mass_fraction_of_snow_in_air_ml
    # mass_fraction_of_cloud_ice_in_air_ml
    # "cloud_base_altitude","cloud_top_altitude"
    # mass_fraction_of_cloud_condensed_water_in_air_ml: Grid

    # dtdz_p = np.where(p > p_top, dmet.dtdz[:, :, jindx, iindx], np.NaN).squeeze()
    # rh_p = relative_humidity(temp_p, q_p, p_p)
    # massfrac_cloud_cond = dmet_ml.mass_fraction_of_cloud_condensed_water_in_air_ml[:, :, jindx, iindx]
    # areafrac_cloud_cond = dmet_ml.cloud_area_fraction_ml[:, :, jindx, iindx

    # conv_cloud = dmet_ml.convective_cloud_area_fraction[:, :, jindx, iindx]
    # "mass_fraction_of_cloud_condensed_water_in_air_ml", "cloud_area_fraction_ml"

    #################################

    print(
        "\n###########################\n"
        "\nINITIALISING PLOTTING: meteogram_vertical \n"
        "\n###########################\n"
    )
    # INITIALISING
    figm1, (axm1, axm2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(wspace=0.001)
    levels = range(len(dmet.hybrid))
    lx, tx = np.meshgrid(levels, dmet.time_normal[:])

    #################################
    # P1: RH with lapserate and BLheight
    #################################
    # Ground color gray
    axm1.fill_between(
        dmet.time_normal[:],
        dmet.surface_air_pressure[:, 0, jindx, iindx] / 100,
        dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100,
        color="gray",
    )
    # spec humidity
    cmap = cm.get_cmap("gnuplot2_r")  # BrBu  BrYlBu
    lvl = np.linspace(np.min(q_p), np.max(q_p), 20)
    CF_Q = axm1.contourf(tx, p_p, q_p, levels=lvl, cmap=cmap, extend="both", zorder=1)

    ticks = np.array([lvl[0], lvl[5], lvl[10], lvl[15], lvl[-1]])

    cbar = nice_vprof_colorbar(
        CF=CF_Q, ax=axm1, ticks=ticks, lvl=lvl, label="Spec. Hum. [g/kg]"
    )

    axm1.contour(
        tx,
        p_p,
        q_p,
        linestyles="dashed",
        levels=ticks[1:-1],
        colors="white",
        linewidth=2,
        zorder=2,
        alpha=0.8,
    )

    axm1.invert_yaxis()
    axm1.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

    #################################
    # P2: Potential temp with wind
    #################################
    # Ground in gray
    axm2.fill_between(
        dmet.time_normal[:],
        dmet.surface_air_pressure[:, 0, jindx, iindx] / 100,
        dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100,
        color="gray",
    )
    cmap = plt.cm.RdYlBu_r  # plt.cm.jet RdYlBu
    skip = (slice(None, None, 2), slice(None, None, 2))

    axm2.barbs(
        tx[skip][skip],
        p_p[skip][skip],
        uml_p[skip][skip] * 1.943844,
        vml_p[skip][skip] * 1.943844,
        length=7,
        zorder=1000,
        sizes=dict(emptybarb=0.25, spacing=0.15, height=0.4),
    )

    lvl = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    CF_WS = axm2.contourf(
        tx, p_p, vel_p, cmap=cmap, alpha=0.8, levels=lvl, extend="both", zorder=1
    )
    axm2.contour(
        tx,
        p_p,
        vel_p,
        levels=[13.0],
        linestyles="dashed",
        colors="black",
        linewidth=2,
        alpha=0.8,
        zorder=1,
    )

    ticks = np.array([3, 13, 20])
    cbar = nice_vprof_colorbar(
        CF=CF_WS,
        ax=axm2,
        ticks=ticks,
        lvl=lvl,
        label="Wind Speed [m/s]",
        highlight_val=[13],
    )

    # potential temp.
    lvl = np.linspace(
        np.min(dmet_ml.theta[:, :, jindx, iindx]),
        np.max(dmet_ml.theta[:, :, jindx, iindx]),
        200,
    )
    CS = axm2.contour(
        tx, p_p, dmet_ml.theta[:, :, jindx, iindx], colors="black", levels=lvl, zorder=2
    )
    axm2.clabel(
        CS,
        [*CS.levels[2:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]],
        inline=True,
        fmt="$\Theta$ = %1.0fK",
    )  # '%1.0fK')
    # label
    axm2.legend([CS], ["Pot. Temp."], loc="upper left").set_zorder(99999)
    axm2.invert_yaxis()
    axm2.set_ylabel("Pressure [hPa]")
    axm2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

    #################################
    # SET ADJUSTMENTS ON AXIS
    #################################
    xfmt_maj = mdates.DateFormatter(
        "%d.%m"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter(
        "%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axm2.xaxis.set_major_locator(mdates.DayLocator())
    axm2.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
    axm2.xaxis.set_major_formatter(xfmt_maj)
    axm2.xaxis.set_minor_formatter(xfmt_min)

    axm1.xaxis.grid(True, which="major", linewidth=2)
    axm1.xaxis.grid(True, which="minor", linestyle="--")
    axm2.xaxis.grid(True, which="major", linewidth=2)
    axm2.xaxis.grid(True, which="minor", linestyle="--")
    axm2.tick_params(axis="x", which="major", pad=12)

    # figm1.tight_layout()
    plt.savefig(
        dirName_b1
        + figname_b1
        + "_LOC"
        + str(ip)
        + "["
        + "{0:.2f}_{1:.2f}]".format(
            dmet.longitude[jindx, iindx], dmet.latitude[jindx, iindx]
        )
        + ".png"
    )
    plt.clf()
    plt.close()

    # INITIALISING
    figm2, (axm1, axm2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(wspace=0.001)
    levels = range(len(dmet.hybrid))
    lx, tx = np.meshgrid(levels, dmet.time_normal[:])

    #################################
    # P1: RH with lapserate and BLheight
    #################################
    # Ground color gray
    axm1.fill_between(
        dmet.time_normal[:],
        dmet.surface_air_pressure[:, 0, jindx, iindx] / 100,
        dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100,
        color="gray",
    )
    cmap = cm.get_cmap("RdYlBu_r")  # BrBu  BrYlBu cool bwr RdYlBu_r
    lvl1 = np.linspace(-15, -9.8, 5)
    lvl2 = np.linspace(-6.5, -0.5, 5)
    lvl3 = np.linspace(0, 10, 5)
    lvl = np.append(lvl1, lvl2)
    lvl = np.append(lvl, lvl3)
    ticks = np.array([-9.8, -6.5, -3, 0, 3, 6])
    norm = mpl.colors.DivergingNorm(vmin=-10.0, vcenter=0.0, vmax=6)
    CF = axm1.pcolormesh(tx, p, dtdz_p, cmap=cmap, zorder=1, norm=norm)  # dtdz_p
    cbar = nice_vprof_colorbar(CF=CF, ax=axm1, ticks=ticks, label="Lapse. rate. [C/km]")
    # relative humidity
    CS = axm1.contour(
        tx, p_p, rh_p, zorder=2, levels=np.arange(0, 100, 10), colors="green"
    )  # Purples BrBu  BrYlBu cool bwr RdYlBu_r
    axm1.clabel(CS, inline=True, fmt="%1.0f")  # '%1.0fK')
    # cloud
    Cfrac = axm1.contourf(
        tx,
        p_p,
        areafrac_cloud,
        hatches=["--", "---"],
        colors="none",
        alpha=0.0,
        levels=[1, 50, 100],
    )
    artists, labels = Cfrac.legend_elements()
    cfrac_leg = axm1.legend(
        artists,
        ["1-50% Cloud cover", " 50-100% Cloud cover"],
        handleheight=2,
        loc="upper left",
    )
    # BLH

    # C_BL = axm1.plot(tx, dmet_ml.BL_p[:, 0, jindx, iindx],colors="k")

    # dmet_ml.BL_p
    # Add the legend manually to the current Axes.
    # ax = plt.gca().add_artist(cfrac_leg)
    # Create another legend for the second line.
    # axm1.legend([CS[0]],["Rel. hum."],loc='lower right')

    # adjust axis
    axm1.invert_yaxis()
    axm1.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)
    # axm1.legend(Cfrac[0], "1%-50% Cloud cover")

    # TEMP
    cmap = cm.get_cmap("twilight_shifted")  # BrBu  BrYlBu
    norm = mpl.colors.DivergingNorm(vmin=-30.0, vcenter=0.0, vmax=20)
    CF_2 = axm2.pcolormesh(tx, p, air_tempC_p, zorder=1, cmap=cmap, norm=norm)  # dtdz_p
    cbar = nice_vprof_colorbar(CF=CF_2, ax=axm2, label="Temp. [K]", extend="both")

    # RH
    C = axm2.contour(
        tx, p, rh_p, zorder=2, levels=np.arange(0, 100, 10), colors="green"
    )  # dtdz_p
    axm2.clabel(C, inline=True, fmt="%1.0f")  # '%1.0fK')

    # adjust axis
    axm2.invert_yaxis()
    axm2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

    # axis
    xfmt_maj = mdates.DateFormatter(
        "%d.%m"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter(
        "%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axm2.xaxis.set_major_locator(mdates.DayLocator())
    axm2.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
    axm2.xaxis.set_major_formatter(xfmt_maj)
    axm2.xaxis.set_minor_formatter(xfmt_min)

    axm1.xaxis.grid(True, which="major", linewidth=2)
    axm1.xaxis.grid(True, which="minor", linestyle="--")
    axm2.xaxis.grid(True, which="major", linewidth=2)
    axm2.xaxis.grid(True, which="minor", linestyle="--")
    axm2.tick_params(axis="x", which="major", pad=12)

    # figm2.tight_layout()
    plt.savefig(
        dirName_b1
        + "2"
        + figname_b1
        + "_LOC"
        + str(ip)
        + "["
        + "{0:.2f}_{1:.2f}]".format(
            dmet.longitude[jindx, iindx], dmet.latitude[jindx, iindx]
        )
        + ".png"
    )
    plt.clf()
    plt.close()


def plot_meteogram(dmet, dmet_sfx, dmet_ml, jindx, iindx, dirName_b0, figname_b0, ip):
    #################################

    print(
        "\n###########################\n"
        "\nINITIALISING PLOTTING: meteogram \n"
        "\n###########################\n"
    )
    figm2, (axm1, axm2, axm3, axm4) = plt.subplots(
        nrows=4, ncols=1, figsize=(12, 14), sharex=True
    )

    def autolabel(rects, axis, fmt="{0:.1f}", space=2):  # for the precip. Got from e
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[::space]:
            height = rect.get_height()
            axis.annotate(
                fmt.format(height).strip("-").strip("0"),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    #################################
    # P1: TEMP and RH and PRECIP
    #################################
    # temp

    T2M = axm1.plot(
        dmet.time_normal,
        dmet.air_temperature_2m[:, -1, jindx, iindx] - 273.15,
        color="brown",
    )
    TML0 = axm1.plot(
        dmet.time_normal,
        dmet.air_temperature_ml[:, -1, jindx, iindx] - 273.15,
        "--",
        color="brown",
    )
    axm1.set_ylabel("Temp ($^\circ$C)", color="brown")
    axm1.tick_params(axis="y", colors="brown")
    TS = axm1.plot(
        dmet.time_normal,
        dmet.air_temperature_0m[:, -1, jindx, iindx] - 273.15,
        "-.",
        color="brown",
    )
    axm1.set_ylabel("Temp ($^\circ$C)", color="brown")
    axm1.tick_params(axis="y", colors="brown")
    min = (
        np.min(
            [
                dmet.air_temperature_ml[:, -1, jindx, iindx],
                dmet.air_temperature_0m[:, -1, jindx, iindx],
            ]
        )
        - 273.15
    )
    max = (
        np.max(
            [
                dmet.air_temperature_ml[:, -1, jindx, iindx],
                dmet.air_temperature_0m[:, -1, jindx, iindx],
            ]
        )
        - 273.15
    )

    axm1.set_ylim(bottom=np.floor(min), top=np.ceil(max))

    axm1_2 = axm1.twinx()
    # RH
    axm1_2.plot(
        dmet.time_normal,
        dmet.relative_humidity_2m[:, 0, jindx, iindx] * 100,
        color="seagreen",
        alpha=0.8,
    )
    axm1_2.set_ylabel(" 2m Rel. Hum. (%)", color="seagreen")
    axm1_2.set_ylim(bottom=0.001, top=100)
    axm1_2.tick_params(axis="y", colors="seagreen")

    # precip
    axm1_3 = axm1.twinx()
    wd = -0.125
    P_bar = axm1_3.bar(
        dmet.time_normal,
        dmet.precip1h[:, 0, jindx, iindx],
        color="blue",
        alpha=0.8,
        width=wd / 4,
        align="edge",
    )
    topidx = 1.0
    maxp = np.nanmax(dmet.precip1h[:, 0, jindx, iindx])
    print(maxp)
    if maxp > topidx:
        topidx = round_up(maxp)
    print(topidx)
    axm1_3.set_ylim(
        bottom=0.001, top=topidx + 0.1 * topidx
    )  # adding 20% of max value for getting some clearense above bar
    autolabel(P_bar, axm1_3, space=1)
    axm1_3.set_yticks([])
    axm1_3.set_xticks([])

    # label
    axm1_3.legend(
        [T2M[0], TML0[0], TS[0], P_bar[0]],
        ["2m Temp.", "ml0 Temp.", "0m Temp", "1h acc precip"],
        loc="upper left",
    ).set_zorder(99999)

    #################################
    # P2: SENSIBLE HEAT, sample size
    #################################
    # specifichum
    Q2M = axm2.plot(
        dmet.time_normal,
        dmet.specific_humidity_2m[:, 0, jindx, iindx] * 1000,
        zorder=2,
        color="green",
        alpha=0.8,
    )
    QML0 = axm2.plot(
        dmet.time_normal,
        dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000,
        "--",
        zorder=2,
        color="green",
        alpha=0.8,
    )
    axm2.set_ylabel("Spec. Hum. (g/kg)", color="green")
    axm2.tick_params(axis="y", colors="green")

    maxq = np.max(dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000)
    topidx = 1.5
    if maxq > 1.5:
        topidx = round_up(maxq, 1)
    axm2.set_ylim(bottom=0, top=topidx + 0.10 * topidx)
    # cloudtypefraclevels.and blh
    # cloud_area_fraction convective_cloud_area_fraction high_type_cloud_area_fraction medium_type_cloud_area_fraction low_type_cloud_area_fraction
    axm2_1 = axm2.twinx()
    # stem

    tot_clf_f = axm2_1.fill_between(
        dmet.time_normal,
        0,
        dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=1,
        color="gray",
        alpha=0.6,
    )
    tot_clf = axm2_1.plot(
        dmet.time_normal,
        dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=1,
        color="k",
    )
    tot_patch = mpl.patches.Patch(color="r", alpha=0.5, linewidth=0)

    conv_clf = axm2_1.plot(
        dmet.time_normal,
        dmet.convective_cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=2,
        color="pink",
    )
    high_clf = axm2_1.plot(
        dmet.time_normal,
        dmet.high_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=2,
        color="lightblue",
        marker=r"$C_H$",
        markersize=12,
    )
    med_clf = axm2_1.plot(
        dmet.time_normal,
        dmet.medium_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=2,
        color="blue",
        marker=r"$C_M$",
        markersize=12,
    )
    low_clf = axm2_1.plot(
        dmet.time_normal,
        dmet.low_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
        zorder=2,
        color="red",
        marker=r"$C_L$",
        markersize=12,
    )
    axm2_1.set_ylabel("Cloud cover %", color="k")
    axm2_1.tick_params(axis="y", colors="k")
    # label
    axm2_1.legend(
        [
            Q2M[0],
            QML0[0],
            (tot_clf[0], tot_patch),
            high_clf[0],
            med_clf[0],
            low_clf[0],
            conv_clf[0],
        ],
        [
            "2m Spec.Hum.",
            "ml0 Spec.Hum.",
            "tot.cloud",
            "hi. cloud",
            "med. cloud",
            "low cloud",
            "conv. cloud",
        ],
        loc="upper left",
    ).set_zorder(99999)
    # axm2.legend([Q2M[0], QML0[0]], ["2m Spec.Hum.", "ml0 Spec.Hum."],
    #            loc='upper left').set_zorder(99999)

    #################################
    # P3: Wind, pressure.
    #################################
    # wind
    wspeed_gust = np.sqrt(
        dmet.ug[:, 0, jindx, iindx] ** 2 + dmet.vg[:, 0, jindx, iindx] ** 2
    )
    wspeed = np.sqrt(dmet.u[:, 0, jindx, iindx] ** 2 + dmet.v[:, 0, jindx, iindx] ** 2)
    GUST = axm3.plot(dmet.time_normal, wspeed_gust, zorder=1, color="magenta")
    WIND = axm3.plot(dmet.time_normal, wspeed, zorder=1, color="darkmagenta")
    axm3.quiver(
        dmet.time_normal,
        wspeed,
        dmet.u[:, 0, jindx, iindx] / wspeed,
        dmet.v[:, 0, jindx, iindx] / wspeed,
        scale=80,
        zorder=2,
    )

    #    autolabel(P_bar, axm1_3, space=1)

    axm3.quiver(
        dmet.time_normal,
        wspeed_gust,
        dmet.ug[:, 0, jindx, iindx] / wspeed_gust,
        dmet.vg[:, 0, jindx, iindx] / wspeed_gust,
        scale=80,
        zorder=2,
    )

    axm3.set_ylabel("wind (m/s)")
    axm3.set_ylim(bottom=0, top=25)
    axm3.tick_params(axis="y", color="darkmagenta")
    # pressure
    axm3_2 = axm3.twinx()
    axm3_2.plot(
        dmet.time_normal,
        dmet.surface_air_pressure[:, 0, jindx, iindx] / 100,
        zorder=1,
        color="k",
    )
    axm3_2.set_ylabel(" Surface Pressure (hPa)")
    # axm3_2.set_ylim(bottom=900, top=1050)
    # label
    axm3_2.legend(
        [GUST[0], WIND[0]], ["10m wind gust", "10m wind (10min mean)"], loc="upper left"
    ).set_zorder(99999)

    #################################
    # P3: FLUXES
    #################################
    P_SH = axm4.plot(
        dmet.time_normal, dmet_sfx.H[:, jindx, iindx], zorder=0, color="blue"
    )
    P_LH = axm4.plot(
        dmet.time_normal, dmet_sfx.LE[:, jindx, iindx], zorder=0, color="orange"
    )
    axm4.set_ylabel("Heat Fluxes (W/m$^2$)")
    # rainfall_amount
    axm4_1 = axm4.twinx()
    tot = (
        dmet.rainfall_amount[:, -1, jindx, iindx]
        + dmet.snowfall_amount[:, -1, jindx, iindx]
        + dmet.graupelfall_amount[:, -1, jindx, iindx]
    )
    rain_frac = ((dmet.rainfall_amount[:, -1, jindx, iindx]) / tot) * 100
    snow_frac = ((dmet.snowfall_amount[:, -1, jindx, iindx]) / tot) * 100
    graupel_frac = ((dmet.graupelfall_amount[:, -1, jindx, iindx]) / tot) * 100

    rain_in = axm4_1.plot(
        dmet.time_normal,
        rain_frac,
        zorder=1,
        color="red",
        linestyle="dashed",
        marker="o",
    )
    snow_in = axm4_1.plot(
        dmet.time_normal,
        snow_frac,
        zorder=1,
        color="gray",
        linestyle="dashed",
        marker="*",
    )
    graupel_in = axm4_1.plot(
        dmet.time_normal,
        graupel_frac,
        zorder=1,
        color="lightblue",
        linestyle="dashed",
        marker="D",
    )
    # tot =  axm4_1.plot(dmet.time_normal, tot, zorder=1,color="k")
    axm4_1.tick_params(axis="y", colors="k")
    axm4_1.set_ylabel("% of instantaneous precip. type")

    # label

    axm4.legend(
        [P_SH[0], P_LH[0], rain_in[0], snow_in[0], graupel_in[0]],
        ["Sensible H.Flux", "Latent H.Flux", "Rain", "Snow", "Graupel"],
        loc="upper left",
    ).set_zorder(99999)

    # axm4.legend([P_SH[0], P_LH[0]], ["Sensible Heat Flux", "Latent Heat Flux"], loc='upper left').set_zorder(99999)

    #################################
    # SET ADJUSTMENTS ON AXIS
    #################################
    xfmt = mdates.DateFormatter(
        "%d.%m\n%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_maj = mdates.DateFormatter(
        "%d.%m"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter(
        "%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axm4.xaxis.set_major_locator(mdates.DayLocator())
    axm4.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
    axm4.xaxis.set_major_formatter(xfmt_maj)
    axm4.xaxis.set_minor_formatter(xfmt_min)

    axm1.xaxis.grid(True, which="major", linewidth=2)
    axm1.xaxis.grid(True, which="minor", linestyle="--")
    axm2.xaxis.grid(True, which="major", linewidth=2)
    axm2.xaxis.grid(True, which="minor", linestyle="--")
    axm3.xaxis.grid(True, which="major", linewidth=2)
    axm3.xaxis.grid(True, which="minor", linestyle="--")
    axm4.xaxis.grid(True, which="major", linewidth=2)
    axm4.xaxis.grid(True, which="minor", linestyle="--")

    axm4.tick_params(axis="x", which="major", pad=12)
    print("before savefig")
    # figm2.tight_layout()
    plt.savefig(
        dirName_b0
        + figname_b0
        + "_LOC"
        + str(ip)
        + "["
        + "{0:.2f}_{1:.2f}]".format(
            dmet.longitude[jindx, iindx], dmet.latitude[jindx, iindx]
        )
        + ".png"
    )

    plt.close()

    print(
        "\n###########################\n"
        "\nDONE meteogram \n"
        "\n###########################\n"
    )


def plot_maplocation(
    dmet,
    data_domain,
    close2point,
    dirName_b2,
    figname_b2,
    sitename="ALL",
    point_lonlat=None,
    all=False,
):
    map_domain = data_domain  # can be none
    xy = [dmet.x[0], dmet.x[-1], dmet.y[0], dmet.y[-1]]
    lon0 = dmet.longitude_of_central_meridian_projection_lambert
    lat0 = dmet.latitude_of_projection_origin_projection_lambert
    parallels = dmet.standard_parallel_projection_lambert

    figm2 = plt.figure(figsize=(12, 14))
    globe = ccrs.Globe(
        ellipse="sphere", semimajor_axis=6371000.0, semiminor_axis=6371000.0
    )
    crs = ccrs.LambertConformal(
        central_longitude=lon0,
        central_latitude=lat0,
        standard_parallels=parallels,
        globe=globe,
    )
    ax = figm2.add_subplot(projection=crs)

    ip = 0
    if all == True:
        allpoints = deepcopy(close2point)
        close2point = [0]
    for p in close2point:
        if all == True:
            p = allpoints
        ax.cla()

        ax.background_patch.set_facecolor("lightskyblue")  # fill_color='lightskyblue'

        lonlat = [
            dmet.longitude[0, 0],
            dmet.longitude[-1, -1],
            dmet.latitude[0, 0],
            dmet.latitude[-1, -1],
        ]
        svalbard_lonlat = [-8, 30, 73, 82]

        if (
            lonlat[0] > svalbard_lonlat[0]
            and lonlat[1] < svalbard_lonlat[1]
            and lonlat[2] > svalbard_lonlat[2]
            and lonlat[3] < svalbard_lonlat[3]
        ):
            print("svalbard setup")
            file_path = "../../data/shapefiles/svalbard/S100_Land_f_WGS84.shp"  # relative path to the file
            shp = shapereader.Reader(
                file_path
            )  # facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2
            ax.add_geometries(
                shp.geometries(),
                crs=ccrs.PlateCarree(),
                facecolor="whitesmoke",
                edgecolor="k",
                linewidths=1.0,
                zorder=2,
            )  # instead of ax.coastline()
        else:
            ax.add_feature(
                cfeature.GSHHSFeature(scale="high"),
                facecolor="whitesmoke",
                edgecolor="k",
                linewidths=1.0,
                zorder=2,
            )  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            # ax.coastlines(resolution='10m')
        # xy = [dmet.x[0], dmet.x[-1], dmet.y[0],dmet.y[-1]] #ax.set_extent((xy[0], xy[1], xy[2], xy[3])) #(x0, x1, y0, y1)
        ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)

        all_gridpoint = ax.plot(
            dmet.longitude,
            dmet.latitude,
            marker=".",
            markersize=6.0,
            markeredgewidth=4,
            transform=ccrs.PlateCarree(),
            markerfacecolor="red",
            markeredgecolor="black",
            zorder=1000,
            linestyle="None",
        )

        # print(point_lonlat[0])
        # print(point_lonlat[1])
        mainpoint = ax.plot(
            point_lonlat[0],
            point_lonlat[1],
            marker=".",
            markersize=5.0,
            markeredgewidth=4,
            transform=ccrs.PlateCarree(),
            markerfacecolor="blue",
            markeredgecolor="blue",
            zorder=1000,
            linestyle="None",
        )

        ax.plot(
            dmet.longitude[p],
            dmet.latitude[p],
            marker=".",
            markersize=6.0,
            markeredgewidth=4,
            transform=ccrs.PlateCarree(),
            markerfacecolor="red",
            markeredgecolor="red",
            zorder=1000,
            linestyle="None",
        )

        CC = plt.contour(
            dmet.longitude,
            dmet.latitude,
            dmet.land_area_fraction[0, 0, :, :],
            alpha=0.6,
            zorder=3,
            levels=[0.9, 1, 1.1],
            colors="b",
            linewidths=5,
            transform=ccrs.PlateCarree(),
        )

        if all == True:
            # ss = ""
            # for x,y in dmet.longitude[p], dmet.latitude[p]:
            #    ss= ss+"[" + "{0:.2f}_{1:.2f}]".format(x, y)
            figname_b2_2 = figname_b2 + "_[" + sitename + "]"
        else:
            figname_b2_2 = (
                figname_b2
                + "_LOC"
                + str(ip)
                + "["
                + "{0:.2f}_{1:.2f}]".format(dmet.longitude[p], dmet.latitude[p])
            )
        # figm2.tight_layout()
        plt.savefig(dirName_b2 + figname_b2_2 + ".png")
        ip += 1


##########################################################
def setup_directory(modelrun, point_name, point_lonlat):
    projectpath = setup_directory(OUTPUTPATH, "{0}".format(modelrun))
    figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        dirName = (
            projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
        )
    else:
        dirName = (
            projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_name)
        )

    dirName_b1 = dirName + "met/"
    figname_b1 = "vmet_" + figname

    dirName_b0 = dirName + "met/"
    figname_b0 = "met_" + figname

    dirName_b2 = dirName + "map/"
    figname_b2 = "map_" + figname

    dirName_b3 = dirName + "met/"
    figname_b3 = "met_" + figname

    if not os.path.exists(dirName_b1):
        os.makedirs(dirName_b1)
        print("Directory ", dirName_b1, " Created ")
    else:
        print("Directory ", dirName_b1, " already exists")
    if not os.path.exists(dirName_b2):
        os.makedirs(dirName_b2)
        print("Directory ", dirName_b2, " Created ")
    else:
        print("Directory ", dirName_b2, " already exists")
    return (
        dirName_b0,
        dirName_b1,
        dirName_b2,
        dirName_b3,
        figname_b0,
        figname_b1,
        figname_b2,
        figname_b3,
    )


##########################################################
def meteogram_average(dmet, dmet_sfx, dmet_ml, indx, dirName_b2, figname_b2, sitename):
    # lona = dmet.longitude[indx[0], indx[1]]
    # lata = dmet.latitude[indx[0], indx[1]]
    figma1, (axma1, axma2, axma3, axma4) = plt.subplots(
        nrows=4, ncols=1, figsize=(12, 14), sharex=True
    )

    def autolabel(rects, axis, fmt="{0:.1f}", space=2):  # for the precip. Got from e
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[::space]:
            height = rect.get_height()
            axis.annotate(
                fmt.format(height).strip("-").strip("0"),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                zorder=99999,
            )

    #################################
    # P1: temp, rh, precip
    #################################
    # temp
    temp2m_mean = np.mean(dmet.air_temperature_2m[:, 0, indx[0], indx[1]], axis=(1))

    T2M_MEAN = axma1.plot(
        dmet.time_normal, temp2m_mean - 273.15, color="red", linewidth=3
    )
    T2M = axma1.plot(
        dmet.time_normal,
        dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15,
        color="red",
        linewidth=0.2,
        alpha=0.7,
    )

    axma1.set_ylabel("Temp ($^\circ$C)", color="red")
    axma1.tick_params(axis="y", colors="red")
    axma1.set_ylim(bottom=-25, top=0)
    min = np.min([dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15])
    max = np.max([dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15])
    axma1.set_ylim(bottom=np.floor(min), top=np.ceil(max))
    #
    # T2M = axm1.plot(dmet.time_normal, dmet.air_temperature_2m[:, -1, jindx, iindx] - 273.15, color="brown")
    # TML0 = axm1.plot(dmet.time_normal, dmet.air_temperature_ml[:, -1, jindx, iindx] - 273.15, "--", color="brown")
    # axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
    # axm1.tick_params(axis="y", colors="brown")
    # TS = axm1.plot(dmet.time_normal, dmet.air_temperature_0m[:, -1, jindx, iindx] - 273.15, "-.", color="brown")
    # axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
    # axm1.tick_params(axis="y", colors="brown")
    # min = np.min([dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15
    # max = np.max([dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15

    # axm1.set_ylim(bottom=np.floor(min), top=np.ceil(max))

    # axm1_2 = axm1.twinx()

    # rh
    axma1_2 = axma1.twinx()
    relhum2m_mean = np.mean(dmet.relative_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
    RH2m_MEAN = axma1_2.plot(
        dmet.time_normal, relhum2m_mean * 100, color="seagreen", linewidth=3
    )
    RH2M = axma1_2.plot(
        dmet.time_normal,
        dmet.relative_humidity_2m[:, 0, indx[0], indx[1]] * 100,
        color="seagreen",
        linewidth=0.2,
        alpha=0.7,
    )
    axma1_2.set_ylabel(" 2m Rel. Hum. (%)", color="seagreen")
    axma1_2.set_ylim(bottom=0.001, top=100)
    axma1_2.tick_params(axis="y", colors="seagreen")
    # precip
    axma1_3 = axma1.twinx()
    wd = -0.125
    precip_mean = np.mean(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
    precip_max = np.max(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
    precip_min = np.min(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
    P_bar_max = axma1_3.bar(
        dmet.time_normal,
        precip_max,
        color="lightblue",
        alpha=0.5,
        width=wd / 4,
        align="edge",
        bottom=0,
        zorder=10,
    )
    P_bar = axma1_3.bar(
        dmet.time_normal,
        precip_mean,
        color="blue",
        alpha=0.5,
        width=wd / 4,
        align="edge",
        bottom=0,
        zorder=11,
    )
    # P_bar_min = axma1_3.bar(time_normal, precip_min, color="red", alpha=1, width=wd / 4, align="edge", bottom = 0, zorder=12)
    autolabel(P_bar_max, axma1_3, space=1)
    # autolabel(P_bar, axma1_3, space=1)
    topidx = 1
    maxp = np.nanmax(precip_max[:])
    if maxp > topidx:
        topidx = round_up(maxp)
    axma1_3.set_ylim(bottom=0.001, top=topidx + 0.2 * topidx)

    axma1_3.set_yticks([])
    axma1_3.legend(
        [T2M_MEAN[0], RH2m_MEAN[0], P_bar[0]],
        ["2m mean Temp.", "2m mean RH", "1h acc mean/max precip"],
        loc="upper left",
    ).set_zorder(99999)
    # xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    # axma1.xaxis.set_major_formatter(xfmt)
    #################################
    # P2: Spec Hum. and cloudtype
    #################################
    q_mean2m = np.mean(dmet.specific_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
    Q2M_mean = axma2.plot(
        dmet.time_normal, q_mean2m * 1000, zorder=0, color="green", alpha=1, linewidth=3
    )
    Q2M = axma2.plot(
        dmet.time_normal,
        dmet.specific_humidity_2m[:, 0, indx[0], indx[1]] * 1000,
        zorder=0,
        color="green",
        alpha=0.7,
        linewidth=0.2,
    )
    axma2.set_ylabel("Spec. Hum. (g/kg)", color="green")
    axma2.tick_params(axis="y", colors="green")
    maxq = np.max(dmet.specific_humidity_ml[:, -1, indx[0], indx[1]] * 1000)
    topidx = 1.5
    if maxq > 1.5:
        topidx = round_up(maxq, 1)
    axma2.set_ylim(bottom=0, top=topidx)

    axma2_1 = axma2.twinx()
    # stem
    all_low_clf = np.mean(
        dmet.low_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1)
    )
    all_mid_clf = np.mean(
        dmet.medium_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1)
    )
    all_high_clf = np.mean(
        dmet.high_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1)
    )
    all_conv_clf = np.mean(
        dmet.convective_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1)
    )
    tot_all = np.mean(dmet.cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))

    tot_clf_f = axma2_1.fill_between(
        dmet.time_normal, 0, tot_all * 100, zorder=1, color="gray", alpha=0.6
    )
    tot_clf = axma2_1.plot(dmet.time_normal, tot_all * 100, zorder=1, color="k")
    tot_patch = mpl.patches.Patch(color="gray", alpha=0.6, linewidth=0)

    conv_clf = axma2_1.plot(
        dmet.time_normal, all_conv_clf * 100, zorder=2, color="pink"
    )
    high_clf = axma2_1.plot(
        dmet.time_normal,
        all_high_clf * 100,
        zorder=2,
        color="lightblue",
        marker=r"$C_H$",
        markersize=12,
    )
    med_clf = axma2_1.plot(
        dmet.time_normal,
        all_mid_clf * 100,
        zorder=2,
        color="blue",
        marker=r"$C_M$",
        markersize=12,
    )
    low_clf = axma2_1.plot(
        dmet.time_normal,
        all_low_clf * 100,
        zorder=2,
        color="red",
        marker=r"$C_L$",
        markersize=12,
    )
    axma2_1.set_ylabel("Cloud cover %", color="k")
    axma2_1.tick_params(axis="y", colors="k")
    # label
    axma2_1.legend(
        [
            Q2M[0],
            (tot_clf[0], tot_patch),
            high_clf[0],
            med_clf[0],
            low_clf[0],
            conv_clf[0],
        ],
        [
            "2m Spec.Hum.",
            "tot.cloud",
            "hi. cloud",
            "med. cloud",
            "low cloud",
            "conv. cloud",
        ],
        loc="upper left",
    ).set_zorder(99999)

    ## label
    # axma2.legend([Q2M_mean[0]], ["2m mean Spec.Hum"],loc='upper left').set_zorder(99999)

    #################################
    # P3: Wind ans pressure
    #################################
    wspeed_gust = np.sqrt(
        dmet.x_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2
        + dmet.y_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2
    )
    wspeed = np.sqrt(
        dmet.x_wind_10m[:, 0, indx[0], indx[1]] ** 2
        + dmet.y_wind_10m[:, 0, indx[0], indx[1]] ** 2
    )
    wsg_mean = np.mean(wspeed_gust, axis=(1))
    ws_mean = np.mean(wspeed, axis=(1))
    WIND_MEAN = axma3.plot(
        dmet.time_normal, ws_mean, zorder=1, color="darkmagenta", linewidth=3, alpha=1
    )
    WIND = axma3.plot(
        dmet.time_normal,
        wspeed,
        zorder=1,
        color="darkmagenta",
        linewidth=0.2,
        alpha=0.7,
    )
    GUST_MEAN = axma3.plot(
        dmet.time_normal, wsg_mean, zorder=0, color="magenta", linewidth=3, alpha=1
    )
    GUST = axma3.plot(
        dmet.time_normal,
        wspeed_gust,
        zorder=0,
        color="magenta",
        linewidth=0.2,
        alpha=0.7,
    )
    axma3.set_ylabel("wind (m/s)")
    axma3.set_ylim(bottom=0, top=25)
    axma3.tick_params(axis="y", color="darkmagenta")

    axma3_3 = axma3.twinx()
    p_mean = np.mean(dmet.surface_air_pressure[:, 0, indx[0], indx[1]], axis=(1))
    PP = axma3_3.plot(dmet.time_normal, p_mean / 100, zorder=1, color="k", linewidth=3)
    axma3_3.set_ylabel(" Surface Pressure (hPa)")

    axma3_3.legend(
        [GUST_MEAN[0], WIND_MEAN[0], PP[0]],
        ["10m wind gust", "10m wind (10min mean)", "mean surf. pressure"],
        loc="upper left",
    ).set_zorder(99999)

    #################################
    # P4: FLux of sensible/latent heat
    #################################
    SH_mean = np.mean(dmet_sfx.H[:, indx[0], indx[1]], axis=(1))
    LH_mean = np.mean(dmet_sfx.LE[:, indx[0], indx[1]], axis=(1))
    P_SH_MEAN = axma4.plot(
        dmet.time_normal, SH_mean, zorder=0, color="blue", linewidth=3, alpha=1
    )
    P_SH = axma4.plot(
        dmet.time_normal,
        dmet_sfx.H[:, indx[0], indx[1]],
        zorder=0,
        color="blue",
        linewidth=0.2,
        alpha=0.7,
    )
    P_LH_MEAN = axma4.plot(
        dmet.time_normal, LH_mean, zorder=0, color="orange", linewidth=3, alpha=1
    )
    P_LH = axma4.plot(
        dmet.time_normal,
        dmet_sfx.LE[:, indx[0], indx[1]],
        zorder=0,
        color="orange",
        linewidth=0.2,
        alpha=0.7,
    )
    axma4.set_ylabel("Fluxes (W/m$^2$)")
    axma4.legend(
        [P_SH_MEAN[0], P_LH_MEAN[0]],
        ["Sensible Heat Flux", "Latent Heat Flux"],
        loc="upper left",
    ).set_zorder(99999)
    # axma4.xaxis.set_major_formatter(xfmt)

    axma4_1 = axma4.twinx()

    tot_rain = np.nansum(dmet.rainfall_amount[:, -1, indx[0], indx[1]], axis=(1))
    tot_snow = np.nansum(dmet.snowfall_amount[:, -1, indx[0], indx[1]], axis=(1))
    tot_graupel = np.nansum(dmet.graupelfall_amount[:, -1, indx[0], indx[1]], axis=(1))

    tot = tot_rain + tot_snow + tot_graupel
    rain_frac = (tot_rain / tot) * 100
    snow_frac = (tot_snow / tot) * 100
    graupel_frac = (tot_graupel / tot) * 100

    rain_in = axma4_1.plot(
        dmet.time_normal,
        rain_frac,
        zorder=1,
        color="red",
        linestyle="dashed",
        marker="o",
    )
    snow_in = axma4_1.plot(
        dmet.time_normal,
        snow_frac,
        zorder=1,
        color="gray",
        linestyle="dashed",
        marker="*",
    )
    graupel_in = axma4_1.plot(
        dmet.time_normal,
        graupel_frac,
        zorder=1,
        color="lightblue",
        linestyle="dashed",
        marker="D",
    )
    axma4_1.tick_params(axis="y", colors="k")
    axma4_1.set_ylabel("% of instantaneous precip. type")
    # label
    axma4_1.legend(
        [P_SH[0], P_LH[0], rain_in[0], snow_in[0], graupel_in[0]],
        ["Sensible H.Flux", "Latent H.Flux", "Rain", "Snow", "Graupel"],
        loc="upper left",
    ).set_zorder(99999)

    #################################
    # SET ADJUSTMENTS ON AXIS
    #################################
    xfmt = mdates.DateFormatter(
        "%d.%m\n%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_maj = mdates.DateFormatter(
        "%d.%m"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter(
        "%HUTC"
    )  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axma4.xaxis.set_major_locator(mdates.DayLocator())
    axma4.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
    axma4.xaxis.set_major_formatter(xfmt_maj)
    axma4.xaxis.set_minor_formatter(xfmt_min)

    axma1.xaxis.grid(True, which="major", linewidth=2)
    axma1.xaxis.grid(True, which="minor", linestyle="--")
    axma2.xaxis.grid(True, which="major", linewidth=2)
    axma2.xaxis.grid(True, which="minor", linestyle="--")
    axma3.xaxis.grid(True, which="major", linewidth=2)
    axma3.xaxis.grid(True, which="minor", linestyle="--")
    axma4.xaxis.grid(True, which="major", linewidth=2)
    axma4.xaxis.grid(True, which="minor", linestyle="--")

    axma4.tick_params(axis="x", which="major", pad=12)

    # figma1.tight_layout()
    plt.savefig(dirName_b2 + figname_b2 + "_LOC[" + sitename + "]" + ".png")
    plt.close()


# def T850_RH(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False, save=True):


def read_data(dt, steps, model, domain_name, domain_lonlat):
    param_ML = [
        "air_temperature_ml",
        "specific_humidity_ml",
        "x_wind_ml",
        "y_wind_ml",
        "mass_fraction_of_cloud_condensed_water_in_air_ml",
        "cloud_area_fraction_ml",
    ]
    param_SFC = [
        "air_temperature_2m",
        "air_temperature_0m",
        "surface_air_pressure",
        "air_pressure_at_sea_level",
        "surface_geopotential",
        "atmosphere_boundary_layer_thickness",
        "relative_humidity_2m",
        "x_wind_gust_10m",
        "y_wind_gust_10m",
        "x_wind_10m",
        "y_wind_10m",
        "specific_humidity_2m",
        "precipitation_amount_acc",
        "land_area_fraction",
        "cloud_base_altitude",
        "cloud_top_altitude",
        "convective_cloud_area_fraction",
        "cloud_area_fraction",
        "high_type_cloud_area_fraction",
        "medium_type_cloud_area_fraction",
        "low_type_cloud_area_fraction",
        "rainfall_amount",
        "snowfall_amount",
        "graupelfall_amount",
    ]
    param = param_SFC + param_ML
    param_sfx = ["SST", "H", "LE", "TS"]
    # cloud_base_altitude  cloud_top_altitude  air_temperature_0m  , "upward_air_velocity_ml"
    # calc relative hum from...? "mass_fraction_of_rain_in_air_ml","mass_fraction_of_graupel_in_air_ml"
    # "mass_fraction_of_cloud_ice_in_air_ml","mass_fraction_of_snow_in_air_ml",

    split = False
    print("\n######## Checking if your request is possible ############")
    try:
        check_all = check_data(date=dt, model=model, param=param, step=steps)
    except ValueError:
        split = True
        try:
            print("--------> Splitting up your request to find match ############")
            check_sfc = check_data(date=dt, model=model, param=param_SFC, step=steps)
            check_ml = check_data(date=dt, model=model, param=param_ML, step=steps)
        except ValueError:
            print(
                "!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!"
            )
            sys.exit(1)
            # break
    print("--------> Found match for your request ############")
    print(check_all.file)
    try:
        check_sfx = check_data(date=dt, model=model, param=param_sfx, step=steps)
    except ValueError:
        param_sfx = ["SFX_SST", "SFX_H", "SFX_LE", "SFX_TS"]
        try:
            check_sfx = check_data(date=dt, model=model, param=param_sfx, step=steps)
        except ValueError:
            print(
                "!!!!! Missing surfex data. Sorry this plot is not availbale for this date. Try with another datetime !!!!!"
            )
            sys.exit(1)
            # break

    if not split:
        file_all = check_all.file.loc[0]
        data_domain = domain_input_handler(
            dt, model, domain_name, domain_lonlat, file_all
        )

        print(data_domain.idx)
        dmet = get_data(
            model=model,
            data_domain=data_domain,
            param=param,
            file=file_all,
            step=steps,
            date=dt,
        )
        print("dmet success")
        file_sfx = check_sfx.file.loc[0]
        dmet_sfx = get_data(
            model=model,
            data_domain=data_domain,
            param=param_sfx,
            file=file_sfx,
            step=steps,
            date=dt,
        )
        print("\n######## Retriving data ############")
        print(f"--------> from: {dmet.url} ")
        dmet.retrieve()
        print("\n######## Retriving data ############")
        print(f"--------> from: {dmet_sfx.url} ")
        dmet_sfx.retrieve()
        dmet_ml = dmet  # two names for same value, no copying done.
    else:
        # get sfc level data
        file_sfc = check_sfc.file.loc[0]
        data_domain = domain_input_handler(
            dt, model, domain_name, domain_lonlat, file_sfc
        )
        # lonlat = np.array(data_domain.lonlat)
        dmet = get_data(
            model=model,
            param=param_sfc,
            file=file_sfc,
            step=steps,
            date=dt,
            data_domain=data_domain,
        )

        file_sfx = check_sfx.file.loc[0]
        dmet_sfx = get_data(
            model=model,
            data_domain=data_domain,
            param=param_sfx,
            file=file_sfx,
            step=steps,
            date=dt,
        )
        print("\n######## Retriving data ############")
        print(f"--------> from: {dmap_meps.url} ")
        dmet.retrieve()

        # get model level data
        file_ml = check_ml.file.loc[0]
        dmet_ml = get_data(
            model=model,
            data_domain=data_domain,
            param=param_pl,
            file=file_ml,
            step=steps,
            date=dt,
        )
        print("\n######## Retriving data ############")
        print(f"--------> from: {tmap_meps.url} ")
        dmet_ml.retrieve()
        print("\n######## Retriving data ############")
        print(f"--------> from: {dmet_sfx.url} ")
        dmet_sfx.retrieve()
    return dmet, dmet_sfx, dmet_ml, data_domain


def calculate_data(dmet, dmet_ml, dmet_sfx):
    dmet.time_normal = timestamp2utc(dmet.time)
    dmet.modelrun = timestamp2utc([dmet.forecast_reference_time])

    dmet.precip1h = precip_acc(dmet.precipitation_amount_acc, acc=1)
    dmet.precip3h = precip_acc(dmet.precipitation_amount_acc, acc=3)
    # future speedup.. maybe do it for only points needed? But units changes as it is used for display later.
    dmet_ml.p = ml2pl(dmet_ml.ap, dmet_ml.b, dmet_ml.surface_air_pressure)
    dmet_ml.theta = potential_temperatur(dmet_ml.air_temperature_ml, dmet_ml.p)
    dmet_ml.specific_humidity_mlgkg = dmet_ml.specific_humidity_ml * 1000.0  # g/kg
    dmet_ml.heighttoreturn = ml2alt_gl(
        air_temperature_ml=dmet_ml.air_temperature_ml,
        specific_humidity_ml=dmet_ml.specific_humidity_ml,
        ap=dmet_ml.ap,
        b=dmet_ml.b,
        surface_air_pressure=dmet.surface_air_pressure,
    )

    dmet_ml.geotoreturn = ml2alt_sl(
        dmet.surface_geopotential,
        air_temperature_ml=dmet_ml.air_temperature_ml,
        specific_humidity_ml=dmet_ml.specific_humidity_ml,
        ap=dmet_ml.ap,
        b=dmet_ml.b,
        surface_air_pressure=dmet.surface_air_pressure,
    )
    dmet_ml.heighttoreturnhalf = ml2alt_gl(
        air_temperature_ml=dmet_ml.air_temperature_ml,
        specific_humidity_ml=dmet_ml.specific_humidity_ml,
        ap=dmet_ml.ap,
        b=dmet_ml.b,
        surface_air_pressure=dmet.surface_air_pressure,
        inputlevel="half",
        returnlevel="full",
    )

    dmet_ml.t_v_level = virtual_temp(
        dmet_ml.air_temperature_ml, dmet_ml.specific_humidity_ml
    )

    dmet_ml.density_ml = density(dmet_ml.t_v_level, dmet.surface_air_pressure)
    dmet_ml.sample_ml = get_samplesize(
        dmet_ml.specific_humidity_ml, dmet_ml.density_ml, acc=3
    )
    dmet.u, dmet.v = xwind2uwind(dmet.x_wind_10m, dmet.y_wind_10m, dmet.alpha)
    dmet_ml.uml, dmet_ml.vml = xwind2uwind(dmet.x_wind_ml, dmet.y_wind_ml, dmet.alpha)
    dmet.ug, dmet.vg = xwind2uwind(
        dmet.x_wind_gust_10m, dmet.y_wind_gust_10m, dmet.alpha
    )
    # dmet_ml.BL_p = alt_gl2pl(dmet.surface_air_pressure,dmet_ml.t_v_level, dmet.atmosphere_boundary_layer_thickness, outshape=None )
    # dmet_ml.BL_p = point_alt_sl2pres_old(jindx, iindx, dmet_ml.BL_p, data_altitude_sl, t_v_level, p, surface_air_pressure,
    #                      surface_geopotential)
    dmet_ml.dtdz = lapserate(
        dmet_ml.air_temperature_ml, dmet_ml.heighttoreturn, dmet.air_temperature_0m
    )
    # dmet_ml.dthetadz = lapserate(dmet_ml.theta, dmet_ml.heighttoreturn)
    # dmet.wspeed = wind_speed(dmet.x_wind_ml, dmet.y_wind_ml)
    # dmet_ml.dudz = lapserate(dmet.wspeed, dmet_ml.heighttoreturn)
    # dmet.Ri = (9.81/dmet_ml.air_temperature_ml)*(dmet_ml.dthetadz/dmet_ml.dudz)

    try:
        dmet_sfx.H = dmet_sfx.SFX_H
        dmet_sfx.LE = dmet_sfx.SFX_LE
        dmet_sfx.SST = dmet_sfx.SFX_SST
        dmet_sfx.TS = dmet_sfx.SFX_TS
    except:
        pass

    return dmet, dmet_ml
    # density_2m = density( t_v_level, dmet.surface_air_pressure)
    # sample_2m = get_samplesize(specific_humidity_ml, density_ml, acc=3)


def meteogram(
    datetime,
    steps=[0, 10],
    model="AromeArctic",
    domain_name="KingsBay_Z1",
    point_name=None,
    point_lonlat=None,
    num_point=1,
    domain_lonlat=None,
    plot="all",
    legend=True,
    info=False,
    save=True,
):  ##python T850_RH.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
    for dt in datetime:
        dmet, dmet_sfx, dmet_ml, data_domain = read_data(
            dt, steps, model, domain_name, domain_lonlat
        )
        dmet, dmet_ml = calculate_data(dmet, dmet_ml, dmet_sfx)

        (
            dirName_b0,
            dirName_b1,
            dirName_b2,
            dirName_b3,
            figname_b0,
            figname_b1,
            figname_b2,
            figname_b3,
        ) = setup_directory(dt, point_name, point_lonlat)
        if point_lonlat:
            ind_list = nearest_neighbour(
                point_lonlat[0],
                point_lonlat[1],
                dmet.longitude,
                dmet.latitude,
                num_point,
            )
        else:
            point = setup_site(point_name)
            ind_list = nearest_neighbour(
                point["lon"], point["lat"], dmet.longitude, dmet.latitude, num_point
            )
            point_lonlat = [point["lon"], point["lat"]]

        plot_maplocation(
            dmet,
            data_domain,
            ind_list[0:num_point],
            dirName_b2,
            figname_b2,
            point_lonlat=point_lonlat,
        )
        ip = 0
        for points in ind_list[0:num_point]:
            jindx, iindx = points
            plot_meteogram_vertical(
                dmet, dmet_sfx, dmet_ml, jindx, iindx, dirName_b1, figname_b1, ip
            )
            plot_meteogram(
                dmet, dmet_sfx, dmet_ml, jindx, iindx, dirName_b0, figname_b0, ip
            )
            ip += 1
        # average plots
        averagesite = [
            "ALL_DOMAIN",
            "ALL_NEAREST",
            "LAND",
            "SEA",
        ]  # "ALL_NEAREST", "LAND", "SEA",
        for sitename in averagesite:
            if sitename == "SEA":
                indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
                meteogram_average(
                    dmet, dmet_sfx, dmet_ml, indx_sea, dirName_b3, figname_b3, sitename
                )
                plot_maplocation(
                    dmet,
                    data_domain,
                    indx_sea,
                    dirName_b2,
                    figname_b2,
                    sitename,
                    point_lonlat,
                    all=True,
                )
            if sitename == "LAND":
                indx_land = np.where(dmet.land_area_fraction[0][0][:][:] == 1)
                meteogram_average(
                    dmet, dmet_sfx, dmet_ml, indx_land, dirName_b3, figname_b3, sitename
                )
                plot_maplocation(
                    dmet,
                    data_domain,
                    indx_land,
                    dirName_b2,
                    figname_b2,
                    sitename,
                    point_lonlat,
                    all=True,
                )
            if sitename == "ALL_NEAREST":
                ll = np.array([list(item) for item in ind_list[0:num_point]])
                jindx = ll[:, 0]
                iindx = ll[:, 1]
                meteogram_average(
                    dmet,
                    dmet_sfx,
                    dmet_ml,
                    [jindx, iindx],
                    dirName_b3,
                    figname_b3,
                    sitename,
                )
                plot_maplocation(
                    dmet,
                    data_domain,
                    [jindx, iindx],
                    dirName_b2,
                    figname_b2,
                    sitename,
                    point_lonlat,
                    all=True,
                )
            if sitename == "ALL_DOMAIN":
                indx_alldomain = np.where(dmet.latitude != None)
                meteogram_average(
                    dmet,
                    dmet_sfx,
                    dmet_ml,
                    indx_alldomain,
                    dirName_b3,
                    figname_b3,
                    sitename,
                )
                # (dmet,data_domain, close2point, dirName_b2, figname_b2, sitename="ALL", point_lonlat=None, all=False
                plot_maplocation(
                    dmet,
                    data_domain,
                    indx_alldomain,
                    dirName_b2,
                    figname_b2,
                    sitename,
                    point_lonlat,
                    all=True,
                )


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
        default=[0, 10],
        nargs="+",
        type=int,
        help="forecast times example --steps 0 3 gives time 0 to 3",
    )
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat",
        default=None,
        nargs="+",
        type=float,
        help="lonmin lonmax latmin latmax",
    )
    parser.add_argument("--point_name", default=None, help="see sites.yaml")
    parser.add_argument(
        "--point_lonlat", default=None, nargs="+", type=float, help="lon lat"
    )
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()
    meteogram(
        datetime=args.datetime,
        steps=args.steps,
        model=args.model,
        domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat,
        legend=args.legend,
        info=args.info,
        num_point=args.point_num,
        point_lonlat=args.point_lonlat,
        point_name=args.point_name,
    )
    # datetime, step=4, model= "MEPS", domain = None
