'''
Containing useful functions
'''

import datetime as dt
import numpy as np
import math

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def potential_temperatur(air_temperature_ml, p):
    p0 = 100000
    theta = air_temperature_ml * (p0 / p) ** 0.286
    return theta

def ml2pl( ap, b, surface_air_pressure ):
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2] #lat in y
    xSize = np.shape(surface_air_pressure)[3] #lon in x
    p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
    for k in range(0,levelSize):
        p[:, k, :, :] = ap[k] + b[k] * surface_air_pressure[:, 0, :, :]
    return p

def ml2alt_sl( p, surface_geopotential, air_temperature_ml, specific_humidity_ml): #or heighttoreturn  #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    # todo: Add formula for this
    Rd = 287.06
    g = 9.80665
    z_h = 0  # why 0?       -->todo

    timeSize, levelSize, ySize, xSize = np.shape(p)
    geotoreturn_m = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    levels = np.arange(0, levelSize)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    p_low = p[:, levelSize - 1, :, :]  # Pa lowest modellecel is 64

    for k in levels_r:
        p_top = p[:, k - 1, :, :]
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

        if k == 0:  # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)  # 0.1 why? -->todo
            alpha = np.log(2)  # 0.3 why?       -->todo
        else:
            dlogP = np.log(np.divide(p_low, p_top))
            dP = p_low - p_top  #positive
            alpha = 1. - ((p_top / dP) * dlogP)

        TRd = t_v_level[:, k, :, :] * Rd
        z_f = z_h + (TRd * alpha)

        geotoreturn_m[:, k, :, :] = z_f + surface_geopotential[:, 0, :, :]
        geotoreturn_m = geotoreturn_m/g

        # update for next level
        z_h = z_h + (TRd * dlogP)
        p_low = p_top

    return geotoreturn_m

def ml2alt_gl(p, air_temperature_ml, specific_humidity_ml ):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209

    # todo: Add formula for this
    Rd = 287.06
    g = 9.80665
    z_h = 0  #

    timeSize, levelSize, ySize, xSize = np.shape(p)
    heighttoreturn = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))


    levels = np.arange(0, levelSize)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    p_low = p[:, levelSize - 1, :, :]  # Pa lowest modellecel is 64

    for k in levels_r:
        p_top = p[:, k - 1, :, :]
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

        if k == 0:  # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)  # 0.1 why? -->todo
            alpha = np.log(2)  # 0.3 why?       -->todo
        else:
            dlogP = np.log(np.divide(p_low, p_top))
            dP = p_low - p_top  #positive
            alpha = 1. - ((p_top / dP) * dlogP)

        TRd = t_v_level[:, k, :, :] * Rd
        z_f = z_h + (TRd * alpha)
        heighttoreturn[:, k, :, :] = z_f / g  # m
        #update for next level
        z_h = z_h + (TRd * dlogP)
        p_low = p_top

    return heighttoreturn

def density(Tv, p):
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    Rd = 287.06
    rho = p/(Rd*Tv)
    #for k in levels_r:
    #    t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])
    return rho

def get_samplesize(q, rho, a=0.5, b = 0.95, acc = 3):
    rho = rho / 1000 #/m3 to /L
    q = q # g/kg
    samplesize = q * rho * a * b * 60 #per hour
    samplesize_acc = np.full(np.shape(samplesize), np.nan)
    for step in range(acc-1, np.shape(samplesize)[0]):
        s_acc = 0
        i = 0
        while i < acc:
            s_acc += samplesize[ step + i - ( acc - 1 ),:,:,:]
            i+=1
        samplesize_acc[step,:,:,:] = s_acc

    return samplesize_acc

def virtual_temp(air_temperature_ml, specific_humidity_ml):
    #todo: adjust so u can send in either multidim array, lesser dim, or just point numbers
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209

    timeSize, levelSize, ySize, xSize = np.shape(air_temperature_ml)
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    levels = np.arange(0, levelSize)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    for k in levels_r:
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

    return t_v_level

def lapserate(air_temperature_ml, heighttoreturn):
    timeSize, levelSize, ySize, xSize = np.shape(air_temperature_ml)

    dt_levels = np.full((timeSize, levelSize, ySize, xSize), np.nan)
    dz_levels = np.full((timeSize, levelSize, ySize, xSize), np.nan)
    dtdz = np.full( (timeSize, levelSize, ySize, xSize), np.nan)
    step = 5 #5 before
    for k in range(0, levelSize - step):
        k_next = k + step
        dt_levels[:, k, :, :] = air_temperature_ml[:, k, :, :] - air_temperature_ml[:, k_next, :,:]  # over -under
        dz_levels[:, k, :, :] = heighttoreturn[:, k, :, :] - heighttoreturn[:, k_next, :, :]  # over -under

    dtdz[:, :, :, :] = np.divide(dt_levels, dz_levels) * 1000  # /km
    return dtdz

def precip_acc(precip, acc=1):
    precipacc = np.full(np.shape(precip), np.nan)
    for t in range(0 + acc, np.shape(precip)[0] ):
        precipacc[t, 0, :, :] = precip[t, 0, :, :] - precip[t - acc, 0, :, :]
        #Set negative values to 0, but I fixed it in plot instead.

    return precipacc


def point_alt_sl2pres(jindx, iindx, point_alt, data_altitude_sl, t_v_level, p, surface_air_pressure, surface_geopotential):
    Rd = 287.06
    g = 9.80665
    timeSize, levelSize = np.shape(p[:, :, jindx, iindx])

    #max index when we have an altitude in our dataset lett or equal to point_altitude
    idx_tk = np.argmax( (data_altitude_sl[:, :, jindx, iindx] <= point_alt[:]), axis=1)
    tv = t_v_level[:, :, jindx, iindx]
    #######################################
    dp = np.zeros(shape=(timeSize, levelSize))
    dlogP = np.zeros(shape=(timeSize, levelSize))
    levels_r = np.arange(0, levelSize)[::-1] # bottom (lvl=64) to top(lvl = 0) of atmos
    p_low = p[:, levelSize - 1, jindx, iindx]  # Pa lowest modellecel is 64
    for k in levels_r:
        p_top = p[:, k - 1, jindx, iindx]

        if k == 0:  # top of atmos, last loop round
            dlogP_p = np.log(p_low / 0.1)  # 0.1 why? -->todo
            alpha = np.log(2)  # 0.3 why?       -->todo
        else:
            dlogP_p = np.log(np.divide(p_low, p_top))
            dP_p = p_low - p_top  # positive
            alpha = 1. - ((p_top / dP_p) * dlogP_p)
        dlogP[:, k] = dlogP_p
        dp[:, k] = dP_p
    ########################################
    for t in range(0, np.shape(tv)[0]):  # 0,1,2
        tv[t, 0:idx_tk[t]] = np.nan
        dp[t, 0:idx_tk[t]] = np.nan
        dlogP[t, 0:idx_tk[t]] = np.nan
    tvdlogP = np.multiply(tv, dlogP)

    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    H = Rd * T_vmean / g  # scale height
    point_alt_gl = point_alt[:, 0] - surface_geopotential[:, 0, jindx, iindx] / g  # convert to height over surface.
    p_point = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(point_alt_gl) / H))

    return p_point

def alt_gl2pres(jindx, iindx, h):
    print("hyb2h")

def timestamp2utc(timestamp):
    time_utc = [dt.datetime.utcfromtimestamp(x) for x in timestamp]
    return time_utc

def xwind2uwind(xwind,ywind, alpha):
    u = xwind.copy()
    v = ywind.copy()
    for t in range(0,np.shape(u)[0]):
        for k in range(0, np.shape(u)[1]):
            u[t,k,:,:] = u[t,k,:,:]  * np.cos(np.deg2rad(alpha[:,:])) - v[t,k,:,:]  * np.sin(np.deg2rad(alpha[:,:]))
            v[t,k,:,:]  = v[t,k,:,:]  * np.cos(np.deg2rad(alpha[:,:])) + u[t,k,:,:]  * np.sin(np.deg2rad(alpha[:,:]))

    return u,v
