#export PYTHONPATH=$PYTHONPATH:/Users/ainajoh/Documents/GitHub/islas/scripts/FC-system
from weathervis.domain import *  # require netcdf4
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
from netCDF4 import Dataset
import os
import datetime

#mydir_new = os.chdir("/Users/ainajoh/Documents/GitHub/islas/weathervis/")

lt = 7
lvl = 64#64#64
modelruntime = "2020031100"#Camp start 20.feb - 14.march
model = "AromeArctic"
xres=1
yres=1
def set_variable2d():
    variable2d_arome = {}
    variable3d_arome = {}
    variable2d_sfx = {}
    resol = 7 #?


    variable2d_arome['surface_air_pressure'] = {}
    variable2d_arome['surface_air_pressure']['name'] = 'SP'
    variable2d_arome['surface_air_pressure']['units'] = 'Pa'
    variable2d_arome['surface_air_pressure']['description'] = 'log of surface pressure'
    variable2d_arome['surface_air_pressure']['precision'] = resol
    variable2d_arome['air_temperature_2m'] = {}
    variable2d_arome['air_temperature_2m']['name'] = 'T2m'
    variable2d_arome['air_temperature_2m']['units'] = 'K'
    variable2d_arome['air_temperature_2m']['description'] = 'Temperature at 2m'
    variable2d_arome['air_temperature_2m']['precision'] = resol
    variable2d_arome['surface_geopotential'] = {}
    variable2d_arome['surface_geopotential']['name'] = 'Zg'
    variable2d_arome['surface_geopotential']['units'] = 'm^2/s^2'
    variable2d_arome['surface_geopotential']['description'] = 'surface geopotential'
    variable2d_arome['surface_geopotential']['precision'] = resol
    variable2d_arome['land_area_fraction'] = {}
    variable2d_arome['land_area_fraction']['name'] = 'LS'
    variable2d_arome['land_area_fraction']['units'] = 'none'
    variable2d_arome['land_area_fraction']['description'] = 'land sea mask'
    variable2d_arome['land_area_fraction']['precision'] = resol
    variable2d_arome['x_wind_10m'] = {}
    variable2d_arome['x_wind_10m']['name'] = 'U_lon_10m'
    variable2d_arome['x_wind_10m']['units'] = 'm/s'
    variable2d_arome['x_wind_10m']['description'] = 'zonal wind at 10m'
    variable2d_arome['x_wind_10m']['precision'] = resol
    variable2d_arome['y_wind_10m'] = {}
    variable2d_arome['y_wind_10m']['name'] = 'V_lat_10m'
    variable2d_arome['y_wind_10m']['units'] = 'm/s'
    variable2d_arome['y_wind_10m']['description'] = 'meriodional wind at 10m'
    variable2d_arome['y_wind_10m']['precision'] = resol
    variable2d_arome['specific_humidity_2m'] = {}
    variable2d_arome['specific_humidity_2m']['name'] = 'Q2m'
    variable2d_arome['specific_humidity_2m']['units'] = 'kg/kg'
    variable2d_arome['specific_humidity_2m']['description'] = 'specific humidity at 2m'
    variable2d_arome['specific_humidity_2m']['precision'] = resol
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time'] = {}
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['name'] = 'SSHF_CUM'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['units'] = 'J.m-2'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['description'] = 'Cum.Sensible heat flux'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['precision'] = resol

    variable2d_sfx['FMU'] = {}
    variable2d_sfx['FMU']['name'] = 'USTRESS'
    variable2d_sfx['FMU']['units'] = 'Kg.m-1.s-1'
    variable2d_sfx['FMU']['description'] = 'Surface wind stress (u)'
    variable2d_sfx['FMU']['precision'] = resol
    variable2d_sfx['FMV'] = {}
    variable2d_sfx['FMV']['name'] = 'VSTRESS'
    variable2d_sfx['FMV']['units'] = 'Kg.m-1.s-1'
    variable2d_sfx['FMV']['description'] = 'Surface wind stress (v)'
    variable2d_sfx['FMV']['precision'] = resol

    variable3d_arome['air_temperature_ml'] = {}
    variable3d_arome['air_temperature_ml']['name'] = 'T'
    variable3d_arome['air_temperature_ml']['units'] = 'K'
    variable3d_arome['air_temperature_ml']['description'] = 'temperature on pressure sigmal levels'
    variable3d_arome['air_temperature_ml']['precision'] = resol  # digit precision
    variable3d_arome['divergence_vertical'] = {}
    variable3d_arome['divergence_vertical']['name'] = 'NH_dW'
    variable3d_arome['divergence_vertical']['units'] = 'm/s * g'
    variable3d_arome['divergence_vertical'][
        'description'] = 'Non Hydrostatic divergence of vertical velocity: D = -g(w(i) -w(i-1))'
    variable3d_arome['divergence_vertical']['precision'] = resol
    variable3d_arome['x_wind_ml'] = {}
    variable3d_arome['x_wind_ml']['name'] = 'U_X'
    variable3d_arome['x_wind_ml']['units'] = 'm/s'
    variable3d_arome['x_wind_ml']['description'] = 'U wind along x axis on pressure sigmal levels'
    variable3d_arome['x_wind_ml']['precision'] = resol
    variable3d_arome['y_wind_ml'] = {}
    variable3d_arome['y_wind_ml']['name'] = 'V_Y'
    variable3d_arome['y_wind_ml']['units'] = 'm/s'
    variable3d_arome['y_wind_ml']['description'] = 'V wind along y axis on pressure sigmal levels'
    variable3d_arome['y_wind_ml']['precision'] = resol
    # variable3d_arome['PRESS.DEPART']={}
    # variable3d_arome['PRESS.DEPART']['name'] = 'NH_dP'
    # variable3d_arome['PRESS.DEPART']['units'] = 'Pa'
    # variable3d_arome['PRESS.DEPART']['description'] = 'NH departure from pressure'
    # variable3d_arome['PRESS.DEPART']['precision'] = 1
    variable3d_arome['specific_humidity_ml'] = {}
    variable3d_arome['specific_humidity_ml']['name'] = 'Q'
    variable3d_arome['specific_humidity_ml']['units'] = 'kg/kg'
    variable3d_arome['specific_humidity_ml']['description'] = 'specific humidity on pressure sigmal levels'
    variable3d_arome['specific_humidity_ml']['precision'] = resol
    variable3d_arome['turbulent_kinetic_energy_ml'] = {}
    variable3d_arome['turbulent_kinetic_energy_ml']['name'] = 'TKE'
    variable3d_arome['turbulent_kinetic_energy_ml']['units'] = 'm^2/s^2'
    variable3d_arome['turbulent_kinetic_energy_ml']['description'] = 'Turbulent kinetic energy on pressure sigmal levels'
    variable3d_arome['turbulent_kinetic_energy_ml']['precision'] = resol
    variable3d_arome['cloud_area_fraction_ml'] = {}
    variable3d_arome['cloud_area_fraction_ml']['name'] = 'CLDFRA'
    variable3d_arome['cloud_area_fraction_ml']['units'] = 'none'
    variable3d_arome['cloud_area_fraction_ml']['description'] = 'cloud fraction'
    variable3d_arome['cloud_area_fraction_ml']['precision'] = 1


    #2dArome
    print("retrive 2darome")
    param2d_arome = [*variable2d_arome.keys()]
    print(param2d_arome)
    print("check 2d arome")
    arome2d = check_data(date=modelruntime, model=model, param=param2d_arome)
    #
    print(arome2d)
    file_arome2d= arome2d.file
    print(arome2d.file)
    #data_domain = DOMAIN(modelruntime, model, file=file_arome2d)
    #data_domain.AromeArctic()
    dmap_arome2d = get_data(model=model, file=file_arome2d, param=param2d_arome, step=[0, lt],date=modelruntime)
    dmap_arome2d.retrieve()
    print(dmap_arome2d.__dir__)
    print("retrive 3darome")
    # 3dArome This can be included in 2darome for timeefficency
    param3d_arome = [*variable3d_arome.keys()]
    arome3d = check_data(date=modelruntime, model=model, param=param3d_arome)
    file_arome3d = arome3d.file

    #data_domain = DOMAIN(modelruntime, model, file=file_arome3d)
    #data_domain.AromeArctic()
    print(file_arome3d)
    dmap_arome3d = get_data(model=model, file=file_arome3d, param=param3d_arome, step=[0, lt],
                        date=modelruntime,  m_level=[0, lvl])
    #print(dmap_arome3d)
    dmap_arome3d.retrieve()
    print(dmap_arome3d.__dir__)
    print("retrive sfxarome")
    #2dsfx
    param2d_sfx = [*variable2d_sfx.keys()]
    sfx2d = check_data(date=modelruntime, model=model, param=param2d_sfx)
    file_sfx2d=sfx2d.file
    #data_domain = DOMAIN(modelruntime, model, file=file_sfx2d)
    #data_domain.AromeArctic()
    dmap_sfx2d = get_data(model=model, file=file_sfx2d, param=param2d_sfx, step=[0, lt],date=modelruntime)
    dmap_sfx2d.retrieve()
    #print(dmap_sfx2d.x ==dmap_arome2d.x)
    #print(len(dmap_arome2d.x))

    #attr
    url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/2020/07/01/arome_arctic_full_2_5km_20200701T18Z.nc?projection_lambert,x,y"
    dataset = Dataset(url)
    attr = {}
    proj = dataset.variables["projection_lambert"]
    #xa = dataset.variables["x"]
    #print(xa.getncattr("_ChunkSizes"))
    #print(type(long(dataset.variables["x"].getncattr("_ChunkSizes")))


    for t in range( 0, len(dmap_arome2d.time )):
        print("Inside for loop")
        output = "./"
        validdate = datetime.datetime(int(modelruntime[0:4]), int(modelruntime[4:6]), int(modelruntime[6:8]), int(modelruntime[8:10])) + datetime.timedelta(hours=t)
        date_time = validdate.strftime("%Y%m%d_%H")
        #flexpart dont like 00, want 24
        if validdate.hour==0:
            dateminus1d=validdate - datetime.timedelta(days=1)
            date_time=dateminus1d.strftime("%Y%m%d") + "_24"
            #d = datetime.today() - timedelta(days=days_to_subtract)

        print(date_time)
        ncid = Dataset(output+ 'AR' +  date_time + '.nc', 'w')
        attr['reference_lon'] = proj.getncattr("longitude_of_central_meridian")
        attr['ydim'] = np.long(len(dmap_arome2d.y[::yres]))#np.long(dataset.variables["y"].getncattr("_ChunkSizes"))  # Use: None
        attr['forecast'] = validdate.strftime("%H")  # 23
        attr['x_resolution'] = np.double("2500.0") #np.double("2500.0")*xres  # Use: None
        attr['center_lon'] = proj.getncattr("longitude_of_central_meridian")
        attr['rotation_radian'] = 0.0
        attr['xdim'] = np.long(len(dmap_arome2d.x[::xres]))#np.long(dataset.variables["x"].getncattr("_ChunkSizes"))  # Use: None
        attr['input_lat'] = proj.getncattr("latitude_of_projection_origin")  # Use: None
        attr['reference_lat'] = proj.getncattr("latitude_of_projection_origin")
        attr['y_resolution'] = np.double("2500.0") #np.double("2500.0")*yres   # Use: None
        attr['date'] =  validdate.strftime("%Y%m%d") # "20180331"
        attr['input_lon'] = proj.getncattr("longitude_of_central_meridian")  # Use: None
        attr['input_position'] = (794.0, 444.0)  # ??  # Use: None
        attr['geoid'] = proj.getncattr("earth_radius") #6370000#6371229.0 #
        attr['center_lat'] = proj.getncattr("latitude_of_projection_origin")

        print("Create netcdf")
        ncid.setncatts(attr)
        x = ncid.createDimension('X', len(dmap_arome2d.x[::xres]))
        y = ncid.createDimension('Y', len(dmap_arome2d.y[::yres]))
        level = ncid.createDimension('level', len(dmap_arome3d.hybrid))
        xs = ncid.createVariable('X', 'i4', ('X',))
        xs.units = 'none'
        xs[:] = range(1, len(dmap_arome2d.x[::xres]) + 1)
        ys = ncid.createVariable('Y', 'i4', ('Y',))
        ys.units = 'none'
        ys[:] = range(1, len(dmap_arome2d.y[::yres]) + 1)

        levels = ncid.createVariable('level', 'i4', ('level',))
        levels.units = 'none'
        levels[:] = range(1,  len(dmap_arome3d.hybrid) + 1 )

        nc_ak = ncid.createVariable('Ak', 'f4', ('level',))
        nc_ak.units = 'none'
        nc_ak[:] = dmap_arome3d.ap
        nc_bk = ncid.createVariable('Bk', 'f4', ('level',))
        nc_bk.units = 'none'
        nc_bk[:] = dmap_arome3d.b

        vid = ncid.createVariable('LON', 'f4', ('Y', 'X'), zlib=True)
        vid.description = 'longitude of the center grid'
        vid[:] = dmap_arome2d.longitude[::xres,::yres]
        vid = ncid.createVariable('LAT', 'f4', ('Y', 'X'), zlib=True)
        vid.description = 'latitude of the center grid'
        vid[:] = dmap_arome2d.latitude[::xres,::yres]
        print(param3d_arome)
        for param in param3d_arome:
            vid = ncid.createVariable(variable3d_arome[param]['name'], 'f4',('level','Y','X'),zlib=True)
            vid.units = variable3d_arome[param]['units']
            vid.description = variable3d_arome[param]['description']
            expressiondata = f"dmap_arome3d.{param}[{t},:,::{xres},::{yres}]"
            data = eval(expressiondata)
            vid[:] = data

        print(param2d_arome)
        for param in param2d_arome:
            vid = ncid.createVariable(variable2d_arome[param]['name'], 'f4', ('Y', 'X'), zlib=True)
            vid.units = variable2d_arome[param]['units']
            vid.description = variable2d_arome[param]['description']
            expressiondata = f"dmap_arome2d.{param}[{t},0,::{xres},::{yres}]"
            data = eval(expressiondata)
            if param =="surface_air_pressure":
                print(param)
                data = np.log(data)
            vid[:] = data
        for param in param2d_sfx:
            vid = ncid.createVariable(variable2d_sfx[param]['name'], 'f4', ('Y', 'X'), zlib=True)
            vid.units = variable2d_sfx[param]['units']
            vid.description = variable2d_sfx[param]['description']
            expressiondata = f"dmap_sfx2d.{param}[{t},::{xres},::{yres}]"
            data = eval(expressiondata)
            vid[:] = data


        ncid.close()

    #return variable2d_arome

variable2d = set_variable2d()
#print(variable2d)






