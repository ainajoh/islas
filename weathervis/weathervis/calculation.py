'''
Containing useful functions
'''

import datetime as dt
import numpy as np
import math


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

def BL_height(atmosphere_boundary_layer_thickness, surface_geopotential):
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

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

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
    theta = temperature * (p0 / pressure) ** (Rd/cp) #[K]
    return theta


def ml2plhalf( ap, b, surface_air_pressure ):
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

def ml2plfull( ap, b, surface_air_pressure):
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

def ml2pl( ap, b, surface_air_pressure ):
    """Metcoop produces netcdf files that has given out ap,b so that p is on full levels imediately:
    Source:
    https://github.com/metno/NWPdocs/wiki/Calculating-model-level-height/_compare/041362b7f5fdc02f5e1ee3dea00ffc9d8d47c2bc...f0b453779e547d96f44bf17803d845061627f7a8

    """
    p = ml2plhalf( ap, b, surface_air_pressure )
    return p


def ml2althalf2full_gl( surface_geopotential, air_temperature_ml, specific_humidity_ml, p=None, ap=None, b=None, surface_air_pressure=None): #or heighttoreturn
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
    if p == None:
        p = ml2plhalf( ap, b, surface_air_pressure )


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

        geotoreturn_m[:, k, :, :] = z_f + surface_geopotential[:, 0, :, :]

        geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k, :, :]/g

        # update for next level
        z_h = z_h + (TRd * dlogP)
        p_low = p_top

    return geotoreturn_m

def ml2althalf2full_sl( air_temperature_ml, specific_humidity_ml,p=None ):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    """
        Parameters
        ----------
        p: [Pa] pressure on each half modellevel: p = ap + b * surface_air_pressure
        surface_geopotential:[m^2/s^2] Surface geopotential (fis)
        air_temperature_ml: [K] temperature on every model level.
        specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

        Returns
        -------
        geotoreturn_m: [m] geopotential height from sealevel on each full modellevel

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
    Rd = 287.06  # [J/kg K] Gas constant for dry air
    g = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level
    if p == None:
        p = ml2plhalf(ap, b, surface_air_pressure)

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
            dlogP = np.log(p_low / 0.1)
            alpha = np.log(2)
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

def ml2alt_gl( surface_geopotential, air_temperature_ml, specific_humidity_ml, p=None, ap=None, b=None, surface_air_pressure=None): #or heighttoreturn
    """
    Parameters
    ----------
    p: [Pa] pressure on each FULL modellevel: p = ap + b * surface_air_pressure
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
    if p == None:
        p = ml2pl( ap, b, surface_air_pressure )

    timeSize, levelSize, ySize, xSize = np.shape(p)
    geotoreturn_m = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    levels = np.arange(0, levelSize)  #index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    p_low = p[:, levelSize - 1, :, :]  # Pa lowest modellecel is 64
    #     geotoreturn_m[:,k,:,:] = geotoreturn_m[:,k+1,:,:] + (Rd * t_v_level[:,k,:,:] / g)* ln(p[:,k+1,:,:] / p[:,k,:,:])


    z_h = 0 # we want value from ground so here lowestmodellevel geopotential is set to 0.
    for k in levels_r: #64, 63, 63
        k_upper = k-1 #61 dont have value for geo. 
        k_lower = k   #62
        k_lowest = k+1 #63
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])


        geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k + 1, :, :] + (Rd * t_v_level[:, k, :, :] / g) * ln(p[:, k + 1, :, :] / p[:, k, :, :])
        #if k == 0:  # top of atmos, last loop round
        #    dlogP = np.log(p_low / 0.1)
        #    alpha = np.log(2)
        #else:
        dlogP = np.log(np.divide(p_low, p_top))
        dP = p_low - p_top  # positive
        heighttoreturn[:, k, :, :] = z_h / g  # m

        TRd = t_v_level[:, k, :, :] * Rd
        # update for next level
        z_h = z_h + (TRd * dlogP)
        p_low = p_top

    return heighttoreturn


def ml2alt_sl( air_temperature_ml, specific_humidity_ml,p=None ):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    """
        Parameters
        ----------
        p: [Pa] pressure on each half modellevel: p = ap + b * surface_air_pressure
        surface_geopotential:[m^2/s^2] Surface geopotential (fis)
        air_temperature_ml: [K] temperature on every model level.
        specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

        Returns
        -------
        geotoreturn_m: [m] geopotential height from sealevel on each full modellevel

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
    Rd = 287.06  # [J/kg K] Gas constant for dry air
    g = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level
    if p == None:
        p = ml2plhalf(ap, b, surface_air_pressure)

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
            dlogP = np.log(p_low / 0.1)
            alpha = np.log(2)
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

def point_alt_sl2pres(jindx, iindx, point_alt, data_altitude_sl, t_v_level, p, surface_air_pressure, surface_geopotential):
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

def alt_gl2pres(jindx, iindx, h):
    print("hyb2h")

def timestamp2utc(timestamp):
    time_utc = [dt.datetime.utcfromtimestamp(x) for x in timestamp]
    return time_utc

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
