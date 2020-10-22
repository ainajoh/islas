'''
Containing useful functions
'''

import datetime as dt
import numpy as np
import math

####################################################################################################################
# UTILITIES
#####################################################################################################################
def nearest_neighbour(plon,plat, longitudes, latitudes, nmin=1):
    """
    Parameters
    ----------
    plon: longitude of a specific location [degrees]
    plat: latitude of a specific location  [degrees]
    longitudes: all longitudes of the model[degrees]
    latitudes: all latitudes of the model  [degrees]
    nmin: number of points you want nearest to your specific location

    Returns
    -------
    indexes as tuples in array for the closest gridpoint near a specific location.
    point = [(y1,x1),(y2,x2)]. This format is done in order to ease looping through points.
    for p in point:
        #gives p = (y1,x1)
        xatlocation = x_wind_10m[:,0,p]
    """
    #source https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    R = 6371.0 #model has 6371000.0
    dlat = np.radians(latitudes - plat) ##lat2 - lat1
    dlon = np.radians(longitudes - plon) #lon2 - lon1
    platm = np.full(np.shape(latitudes), plat)
    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(plat)) * np.cos(np.radians(latitudes)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    dsort = np.sort(d,axis=None)
    closest_idx = np.where(np.isin(d,dsort[0:nmin]))

    point = [(x,y) for x,y in zip(closest_idx[0],closest_idx[1])]


    return point
def get_samplesize(q, rho, a=0.5, b = 0.95, acc = 3):
    """
    Estimating sample size on field campaign
    Parameters
    ----------
    q   [Specific humidity kg/kg]
    rho [density [kg/m^3]]
    a   [Andrew Seidl provided this factor]
    b   [Andrew Seidl provided this factor]
    acc [x hour acc precip]

    Returns
    -------
    DOUBLE CHECK ALL THE UNITS AFTER UPDATE BEFORE USE.
    """
    rho = rho / 1000 #/m3 to /L
    q = q # g/g to g/kg
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
def precip_acc(precip, acc=1):
    """

    Parameters
    ----------
    precip: model precip that is accumulating with respect to the forecast
    acc: The prefered precip accumulation

    Returns
    -------
    precip [ mm / acc hours]
    """
    precipacc = np.full(np.shape(precip), np.nan)
    #precipacc = np.zeros(np.shape(precip))
    for t in range(0 + acc, np.shape(precip)[0] ):
        precipacc[t, 0, :, :] = precip[t, 0, :, :] - precip[t - acc, 0, :, :]
        #Set negative values to 0, but I fixed it in plot instead.
    #precipacc = np.ma.masked_where(precipacc ==np.nan, precipacc)

    return precipacc
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
def timestamp2utc(timestamp):
    time_utc = [dt.datetime.utcfromtimestamp(x) for x in timestamp]
    return time_utc
####################################################################################################################
# THERMODYNAMICS
#####################################################################################################################
def potential_temperatur(temperature, pressure):
    """
    Parameters
    ----------
    temperature [K]
    pressure    [Pa]

    Returns
    -------
    Potential temperature [K]
    """
    p0 = 100000  #[Pa]
    Rd = 287.05  #[J/kg K] Gas constant for dry air
    cp = 1004.  #[J/kg] specific heat for dry air (WH)
    theta = np.full(np.shape(temperature), np.nan)
    for i in range(0,len(pressure)):
        theta[:,i,:,:] = temperature[:,i,:,:]  * (p0 / pressure[i]) ** (Rd/cp) #[K]
    return theta
def density(Tv, p):
    """
    Parameters
    ----------
    Tv: [K]: Virual temp
    p: Pressure on FULL MODELLEVELS, on pressure levels, or surface.

    Returns
    -------
    rho: [kg/m^3]
    """
    Rd = 287.06       #[J/kg K] Gas constant for dry air
    rho = p/(Rd*Tv)
    #for k in levels_r:
    #    t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])
    return rho
def virtual_temp(air_temperature_ml, specific_humidity_ml):
    """

    Parameters
    ----------
    air_temperature_ml [K]
    specific_humidity_ml [kg/kg]

    Returns
    -------
    Vituel temp [K] on full modellevels.
    """
    #todo: adjust so u can send in either multidim array, lesser dim, or just point numbers
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209

    timeSize, levelSize, ySize, xSize = np.shape(air_temperature_ml)
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    levels = np.arange(0, levelSize)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    Rd = 287.06
    for k in levels_r:
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

    return t_v_level
def lapserate(air_temperature_ml, heighttoreturn):
    """
    AINA:todo IDEA look at the diana code for comparison. They make dt/dz, but from specific arome files vc I think.
    NB! understand before use. This takes dt and dz over some define modelstep.
    Since each modellevels is further apart higher up it means its courser and courser defined.
    It is only used on low levels where dz is small, but still there is differences in dz.
    Parameters
    ----------
    air_temperature_ml [K] temp on full modelelevl
    heighttoreturn: [m]: height on full model levels from ground / sea level: does not matter since we are after dz

    Returns
    -------
    lapserate [K/km].
    """
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

####################################################################################################################
# HEIGHT HANDLING
#####################################################################################################################
#model levels to pressure levels
def ml2pl_full2full( ap, b, surface_air_pressure ):
    """
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]

    Returns
    -------
    p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    p [Pa] pressure on each half modellevel.
    """
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2] #lat in y
    xSize = np.shape(surface_air_pressure)[3] #lon in x
    p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
    for k in range(0,levelSize):
        p[:, k, :, :] = ap[k] + b[k] * surface_air_pressure[:, 0, :, :]
    return p
def ml2pl_half2full( ap, b, surface_air_pressure):
    """
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]

    Returns
    -------
    Pressure [Pa] on full modellevels
    Source:
    ------
    #Equations
        Simmons, A. J. and Burridge, D. M. (1981)
    #arome setup
        https://journals.ametsoc.org/waf/article/32/2/609/40089/AROME-MetCoOp-A-Nordic-Convective-Scale
    #Proof that ecmwf uses this equation even though  Simmons, A. J. and Burridge, D. M. (1981) said its bad on upperlevels
        https://confluence.ecmwf.int/pages/viewpage.action?pageId=85405371
    #Gave boundaries P(SURFACE) AND P(TOP)
        https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/d3_vert.pdf
    """
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2]  # lat in y
    xSize = np.shape(surface_air_pressure)[3]  # lon in x
    pfull = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    phalf = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    for k in range(0, levelSize):
        phalf[:, k, :, :] = ap[k] + b[k] * surface_air_pressure[:, 0, :, :]
        if k==0 or  k==levelSize-1: #top level(0), surface(levelSize=64)
            pfull[:, k, :, :]=phalf[:, k, :, :]
        else: #from k=1....to 63
            pfull[:, k, :, :] = 0.5*( phalf[:, k-1, :, :] + phalf[:, k, :, :] )
    return pfull
def ml2pl_half2half( ap, b, surface_air_pressure ):
    """
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]

    Returns
    -------
    p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    p [Pa] pressure on each half modellevel.
    """
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2] #lat in y
    xSize = np.shape(surface_air_pressure)[3] #lon in x
    p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
    for k in range(0,levelSize):
        p[:, k, :, :] = ap[k] + b[k] * surface_air_pressure[:, 0, :, :]
    return p
def ml2pl_full2half( ap, b, surface_air_pressure ):
    """
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]

    Returns
    -------
    p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    p [Pa] pressure on each half modellevel.
     Ah(k-1) = 2*Af(k) - Ah(k)
     Bunnen i halvnivåer er Ps, da må Ah(65) = 0 og Bh(65) = 1
    """
    levels = np.arange(0, 64)  # index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2]  # lat in y
    xSize = np.shape(surface_air_pressure)[3]  # lon in x

    ah = np.zeros(shape=np.shape(ap))
    bh = np.zeros(shape=np.shape(b))
    ph = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    ah[64] = 0
    bh[64] = 1
    ph[:,64,:,:] = surface_air_pressure[:, 0, :, :]
    for k in levels_r:
        print(k)
        ah[k-1]= 2*ap[k]/101320 - ah[k]
        bh[k-1]= 2*b[k] - bh[k]
        print(ah[k-1])
        print(bh[k - 1])
        print(b[k])
        #pfull[:, k, :, :] = 0.5*( phalf[:, k-1, :, :] + phalf[:, k, :, :] )

        ph[:,k-1,:,:]= ah[k-1]*101320 + bh[k-1]*surface_air_pressure[:, 0, :, :]

    return ph





    print("Not implemented yet")
def ml2pl( ap, b, surface_air_pressure, inputlevel="full", returnlevel="full"):
    """
    Check if pressure is on half or full levels, and calls the appropriate function for this.

    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]
    inputlevel = "full" or "half": if input data is on full or half levels
    returnlevel="full" or "half" : if return data should be on full or half levels.

    Returns
    -------
    Pressure levels on full levels if returnlevel="full"
    or
    Pressure levels on half levels if returnlevel="half"

    INFO:
    -------
    Model: Metcoop produces netcdf files that has given out ap,b so that p is on full levels imediately: See email.
    Source: https://github.com/metno/NWPdocs/wiki/Calculating-model-level-height/_compare/041362b7f5fdc02f5e1ee3dea00ffc9d8d47c2bc...f0b453779e547d96f44bf17803d845061627f7a8
    """
    if inputlevel=="full" and returnlevel=="full":
        p = ml2pl_full2full( ap, b, surface_air_pressure)
    elif inputlevel=="half" and returnlevel=="half":
        p = ml2pl_half2half(ap, b, surface_air_pressure)
    elif inputlevel=="half" and returnlevel=="full":
        p = ml2pl_half2full(ap, b, surface_air_pressure)
    elif inputlevel=="full" and returnlevel=="half":
        p = ml2pl_full2half(ap, b, surface_air_pressure)

    return p

#model levels to geopotential height
def pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p): #or heighttoreturn
    """
    Parameters
    ----------
    p: [Pa] pressure on each half modellevel: p = ap + b * surface_air_pressure
    surface_geopotential:[m^2/s^2] Surface geopotential (fis)
    air_temperature_ml: [K] temperature on every model level.
    specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

    Returns
    -------
    geotoreturn_m: [m] geopotential height from groundlevel on each full modellevel

    Calculations:
     ------------------
     The Euler equations are formulated in a terrain-following pressure-based sigma-coordinate system (Simmons and Burridge 1981)
     U,V,T are on full levels, while p on half levels.

     Using hydpsometric equation, but in pressure-based sigma-coordinate system system following (Simmons and Burridge 1981):
     ###################################################################################################################
     * Psi(full_level) = Psi(top half_level) - alpha(full levels)*Rd*Tv
        *Psi(top half_level) = Psi(lower half level) -RdTvln( p(top half_level)/p(lower half level))
        *alpha(full levels) = 1-  p(top half_level)/ (p(lower half level)- p(top half_level)) * ln( p(top half_level)/p(lower half level))
    ###################################################################################################################
    Sources:
    ------------------
    #Arome config
    https://www.researchgate.net/publication/313544695_The_HARMONIE-AROME_model_configuration_in_the_ALADIN-HIRLAM_NWP_system
    #Equation Source#:
    Simmons, A. J. and Burridge, D. M. (1981).
     An energy and angular momentum conserving vertical finite difference scheme and hybrid vertical coordinates.
      Mon. Wea. Rev., 109, 758–766.
    #Implementation source#:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    """
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    #https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
    #http://www.dca.ufcg.edu.br/mna/Anexo-MNA-modulo02b.pdf
    #https://www.ecmwf.int/sites/default/files/elibrary/1981/12284-energy-and-angular-momentum-conserving-finite-difference-scheme-hybrid-coordinates-and-medium.pdf

    Rd = 287.06 #[J/kg K] Gas constant for dry air
    g = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level
    #if p == None:
    #    p = ml2plhalf( ap, b, surface_air_pressure )


    timeSize, levelSize, ySize, xSize = np.shape(p)
    geotoreturn_m = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    levels = np.arange(0, levelSize)  #index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]           #index of heighlevels from bottom(lvl=64) to top(lvl = 0)
    p_low = p[:, levelSize - 1, :, :] # Pa lowest modelcell is 64

    for k in levels_r:                #loops through all levels from bottom to top #64,63,63.....3,2,1,0 #
        p_top = p[:, k - 1, :, :]     #Pressure at the top of that layer
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

        if k == 0:  # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(np.divide(p_low, p_top))
            dP = p_low - p_top  #positive
            alpha = 1. - ((p_top / dP) * dlogP)

        TRd = t_v_level[:, k, :, :] * Rd
        z_f = z_h + (TRd * alpha)

        geotoreturn_m[:, k, :, :] = z_f #+ surface_geopotential[:, 0, :, :]

        geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k, :, :]/g

        # update for next level
        z_h = z_h + (TRd * dlogP)
        p_low = p_top

    return geotoreturn_m
def pl2alt_full2half_gl( air_temperature_ml, specific_humidity_ml, p): #or heighttoreturn
    print("not implemented yet")
def pl2alt_full2full_gl( air_temperature_ml, specific_humidity_ml,p): #or heighttoreturn
    """
    Parameters
    ----------
    p: [Pa] pressure on each FULL modellevel:
    surface_geopotential:[m^2/s^2] Surface geopotential (fis)
    air_temperature_ml: [K] temperature on every model level.
    specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

    Returns
    --------

   ###################################################################################################################
    Sources:
    ------------------
    https://github.com/metno/NWPdocs/wiki/Calculating-model-level-height
    """

    Rd = 287.06 #[J/kg K] Gas constant for dry air
    g = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level

    timeSize, levelSize, ySize, xSize = np.shape(p)
    geotoreturn_m = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    levels = np.arange(0, levelSize-1)  #index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    p_lowf = p[:, levelSize-1, :, :]  # Pa lowest modellecel is 64
    t_v_level= virtual_temp(air_temperature_ml, specific_humidity_ml)
    #     geotoreturn_m[:,k,:,:] = geotoreturn_m[:,k+1,:,:] + (Rd * t_v_level[:,k,:,:] / g)* ln(p[:,k+1,:,:] / p[:,k,:,:])

    for k in levels_r: #64, 63, 63
        p_topf = p[:, k - 1, :, :]     #Pressure at the top of that layer63

        p_top= (p_lowf-p_topf)/np.log(p_lowf/p_topf)
        p_low=p_lowf
        tv_top = t_v_level[:, k - 1, :, :]
        tv_low= t_v_level[:, k , :, :]

        if k == 0:  # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(np.divide(p_low, p_top))
            tv = tv_top
            TRd = tv * Rd
            dP = p_low - p_top
            alpha = 1. - ((p_top / dP) * dlogP)
        #for t in range(0, np.shape(t_v_level)[0]):  # 0,1,2
        #    t_v_level[t, 0:idx_tk[t]] = np.nan
        #    dp[t, 0:idx_tk[t]] = np.nan
        #    dlogP[t, 0:idx_tk[t]] = np.nan

        #tvdlogP = np.multiply(tv_low, alpha)
        #T_vmean = np.divide(np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1))
        #H = Rd * T_vmean / g  # scale height
        #p_low = pp[:, levelSize - 1:, :, :]
        #p_top = pp[:, levelSize - 2:, :, :]
        #z_f = z_h + H * np.log(p_low / p_top)


        z_f = z_h + (TRd * alpha)

        #psi_lower = geotoreturn_m[:, k + 1, :, :] + (TRd * dlogP)

        geotoreturn_m[:, k, :, :] = z_f
        geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k, :, :] / g

        z_h = z_f#z_h + (TRd * dlogP)
        p_low = p_top

    #geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k, :, :]/g #convert to meter
    return geotoreturn_m
def pl2alt_half2half_gl( air_temperature_ml, specific_humidity_ml,p): #or heighttoreturn
    """
    Parameters
    ----------
    p: [Pa] pressure on each FULL modellevel:
    surface_geopotential:[m^2/s^2] Surface geopotential (fis)
    air_temperature_ml: [K] temperature on every model level.
    specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

    Returns
    --------

   ###################################################################################################################
    Sources:
    ------------------
    https://github.com/metno/NWPdocs/wiki/Calculating-model-level-height
    """
    geotoreturn_m = pl2alt_full2full_gl(air_temperature_ml, specific_humidity_ml, p)
    return geotoreturn_m

def ml2alt_gl( air_temperature_ml, specific_humidity_ml, ap, b, surface_air_pressure, inputlevel="full", returnlevel="full"):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209

    if inputlevel == "full" and returnlevel == "full":
        p     = ml2pl_full2full( ap, b, surface_air_pressure )
        gph_m = pl2alt_full2full_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "half" and returnlevel == "half":
        p     = ml2pl_half2half(ap, b, surface_air_pressure)
        gph_m = pl2alt_half2half_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "half" and returnlevel == "full":
        p     = ml2pl_half2full( ap, b, surface_air_pressure )

        gph_m = pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "full" and returnlevel == "half":
        p     = ml2pl_full2half( ap, b, surface_air_pressure )
        gph_m = pl2alt_full2full_gl( air_temperature_ml, specific_humidity_ml, p )

        #gph_m = pl2alt_full2half_gl( air_temperature_ml, specific_humidity_ml, p)

    return gph_m
def ml2alt_sl( surface_geopotential, air_temperature_ml, specific_humidity_ml, ap, b, surface_air_pressure, inputlevel="full", returnlevel="full"):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    gph_m_gl = ml2alt_gl(air_temperature_ml, specific_humidity_ml, ap, b, surface_air_pressure, inputlevel, returnlevel)
    gph_m_sl = gph_m_gl + surface_geopotential
    return gph_m_sl
def pl2alt_gl( air_temperature_ml, specific_humidity_ml, p, inputlevel="full", returnlevel="full"):
    if inputlevel == "full" and returnlevel == "full":
        gph_m = pl2alt_full2full_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "half" and returnlevel == "half":
        gph_m = pl2alt_half2half_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "half" and returnlevel == "full":
        gph_m = pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p )
    elif inputlevel == "full" and returnlevel == "half":
        gph_m = pl2alt_full2half_gl( air_temperature_ml, specific_humidity_ml, p)
    return gph_m
def pl2alt_sl( surface_geopotential, air_temperature_ml, specific_humidity_ml, p, inputlevel="full", returnlevel="full"):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    g = 9.80665
    gph_m_gl = pl2alt_gl(air_temperature_ml, specific_humidity_ml, p, inputlevel, returnlevel)
    gph_m_sl = gph_m_gl + surface_geopotential/g
    return gph_m_sl

#ground level to sealevel and vicaverca
def gl2sl(surface_geopotential, gph_m_gl):
    g = 9.80665
    gph_m_sl = gph_m_gl + surface_geopotential/g
    return gph_m
def sl2gl(surface_geopotential, gph_m_sl):
    g = 9.80665
    gph_m_gl = gph_m_sl - surface_geopotential / g
    return gph_m_gl

#altitude to pressure level
def alt_gl2pl(surface_air_pressure,t_v_level, alt_gl, outshape=None ):
    Rd = 287.06
    g = 9.80665

    #if type(alt_gl)==float or type(alt_gl) == int or type(alt_gl)==str:#if height is constant with time
    #    alt_gl = np.full(np.shape(data_altitude_sl[:, :, jindx, iindx]), float(alt_gl))
    #else: #if height is changes with time it comes as a array or list
    #    alt_gl = np.repeat(point_alt, repeats=len(data_altitude_sl[0, :, jindx, iindx]), axis=0).reshape(np.shape(data_altitude_sl[:, :, jindx, iindx]))

    tvdlogP = np.multiply(tv, dlogP)
    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    H = Rd * T_vmean / g  # scale height
    pl = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(alt_gl) / H))
    return pl

def alt_sl2pl(surface_air_pressure, alt_sl ):
    Rd = 287.06
    g = 9.80665

    tvdlogP = np.multiply(tv, dlogP)
    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    point_alt_gl = alt_sl[:, 0] - surface_geopotential[:, 0, jindx, iindx] / g  # convert to height over surface.
    H = Rd * T_vmean / g  # scale height
    pl = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(point_alt_gl) / H))
    return pl

def point_alt_sl2pres_old(jindx, iindx, point_alt, data_altitude_sl, t_v_level, p, surface_air_pressure, surface_geopotential):
    """
    Converts height from sealevel to pressure.
    Parameters
    ----------
    jindx
    iindx
    point_alt
    data_altitude_sl
    t_v_level
    p
    surface_air_pressure
    surface_geopotential

    Returns
    -------
    """
    Rd = 287.06
    g = 9.80665

    if type(point_alt)==float or type(point_alt) == int or type(point_alt)==str:#if height is constant with time
        point_alt = np.full(np.shape(data_altitude_sl[:, :, jindx, iindx]), float(point_alt))
    else: #if height is changes with time it comes as a array or list
        point_alt = np.repeat(point_alt, repeats=len(data_altitude_sl[0, :, jindx, iindx]), axis=0).reshape(np.shape(data_altitude_sl[:, :, jindx, iindx]))

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


#Special height calc
def BL_height_sl(atmosphere_boundary_layer_thickness, surface_geopotential):
    """
    Parameters
    ----------
    atmosphere_boundary_layer_thickness: Height of the PBL [m] over sealevel
    surface_geopotential: Surface geopotential (fis):  [m^2/s^2]
    Returns
    -------
    BL height over surface
    """
    g=9.80665 #m/s^2
    hgl = atmosphere_boundary_layer_thickness #groundlevel
    hsl = hgl + (surface_geopotential / g)    #
    return hsl




####################################################################################################################
# WIND HANDLING
#####################################################################################################################
# The wind calculations are validated with the use of parameters:
# x_wind_10m, y_wind_10m, wind_speed(height3=10m), wind_direction(height3=10m)
#   u10, v10  = xwind2uwind(tmap_meps.x_wind_10m, tmap_meps.y_wind_10m, tmap_meps.alpha)
#   wsfromuv = wind_speed(u10,v10)
#   wsfromxy = wind_speed(tmap_meps.x_wind_10m, tmap_meps.y_wind_10m)
#   wdfromuv = (np.pi/2 - np.arctan2(v10,u10) + np.pi)*180/np.pi %360
#   wdfromxy =  wind_dir(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m,tmap_meps.alpha)
# Result is true with approx deviaton error of 0.002 or less.
# wsfromuv[0,0,0,0] == wsfromxy[0,0,0,0] == wind_speed[0,0,0,0]
# wdfromuv[0,0,0,0] == wdfromxy[0,0,0,0] == wind_direction[0,0,0,0]
#todo: AINA: alpha: for changing domain, for changing height?
#####################################################################################################################
def xwind2uwind( xwind, ywind, alpha ):
    #source: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    #source: https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    u = np.zeros(shape=np.shape(xwind))
    v = np.zeros(shape=np.shape(ywind))
    for t in range(0,np.shape(xwind)[0]):
        for k in range(0, np.shape(xwind)[1]):
            absdeg2rad = np.abs((alpha)*np.pi/180)
            u[t, k, :, :] = xwind[t, k, :, :] * np.cos(absdeg2rad[:,:]) - ywind[t, k, :, :] * np.sin(absdeg2rad[:,:])
            v[t, k, :, :] = ywind[t, k, :, :] * np.cos(absdeg2rad[:,:]) + xwind[t, k, :, :] * np.sin(absdeg2rad[:,:])

    return u,v

def wind_speed(xwind,ywind):
    #no matter if in modelgrid or earthrelativegrid
    ws = np.sqrt(xwind**2 + ywind**2)
    return ws

def wind_dir(xwind,ywind, alpha):
    #source: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    #https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    #https://stackoverflow.com/questions/21484558/how-to-calculate-wind-direction-from-u-and-v-wind-components-in-r

    wdir = np.zeros(shape=np.shape(xwind))
    for t in range(0,np.shape(wdir)[0]):
        for k in range(0, np.shape(wdir)[1]):
                    #websais:
                    #wdir[t,k,:,:] =  alpha[:,:] + 90 - np.arctan2(ywind[t,k,:,:],xwind[t,k,:,:])
                    #Me:
                    a = ( np.arctan2(ywind[t,k,:,:],xwind[t,k,:,:])*180/np.pi) #mathematical wind angle in modelgrid pointing with the wind
                    b = a + 180  # mathematical wind angle pointing where the wind comes FROM
                    c = 90 - b   # math coordinates(North is 90) to cardinal coordinates(North is 0).
                    wdir[t,k,:,:] =  c + alpha[:,:] #add rotation of modelgrid(alpha).
                    wdir = wdir %360 #making sure is between 0 and 360 with Modulo
    return wdir
