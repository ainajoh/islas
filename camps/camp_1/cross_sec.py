import netCDF4
import numpy as np
import matplotlib.pyplot as plt                 #For basic plotting in python

fig2, ax2 = plt.subplots(figsize=(10, 9))

def makecrosslines(center):
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_extracted_2_5km_latest.nc"
    dataset = netCDF4.Dataset(url)
    x = dataset.variables["x"]
    y = dataset.variables["y"]
    d=5*1000 #km to m
    theta = 0
    x0 = center[0]
    y0 = center[1]

    x1 = x0 + d*cos(theta)  
    y1 = y0 + d*sin(theta)

    x2 = x0 - d * cos(theta)
    y2 = y0 - d * sin(theta)

    #line = #all points betweel x1,y1 and x2,y2 through x0,y0






def main():
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc"
    old_pier = [11.91929, 78.93030]  # E,N like x,y.
    #makecrosslines(old_pier)
    
main()

def get_data():

    dataset = netCDF4.Dataset(url)

    #class regions:
    #    def svalbard(self):
    #p_pl = dataset.variables["pressure"]  # [  50.  100.  150.  200.  250.  300.  400.  500.  700.  800.  850.  925. 1000.]
    #p_level = np.where(p_pl[:] == 850)[0]

    #temp_pl = dataset.variables["air_temperature_pl"][0,p_level,1,1]  #current shape = (67, 13, 949, 739)
    #p_sfc = dataset.variables["surface_air_pressure"][0,0,1,1]  #(67, 1, 949, 739)

def info():
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc"
    dataset = netCDF4.Dataset(url)
    print(dataset)
    #temp_pl = dataset.variables["air_temperature_pl"]
    #p_sfc = dataset.variables["surface_air_pressure"]
    #p_pl = dataset.variables["pressure"]  # [  50.  100.  150.  200.  250.  300.  400.  500.  700.  800.  850.  925. 1000.]

    print(dataset.variables["land_area_fraction"][:]) #(time,height0,y,x)(time,height0,y,x)
    #print(dataset.variables["integral_of_surface_downward_latent_heat_evaporation_flux_wrt_time"]) #(time,height0,y,x)
    #print(dataset.variables["integral_of_surface_downward_latent_heat_sublimation_flux_wrt_time"]) #time,height0,y,x)
    #print(dataset.variables["integral_of_surface_downward_sensible_heat_flux_wrt_time"]) #(time,height0,y,x)
    #print(dataset.variables["x_wind_gust_10m"]) #(time,height0,y,x)

    #integral_of_surface_downward_sensible_heat_flux_wrt_time(time, height0, y, x)




    print(dataset.variables["hybrid"])  #[27315]

    #print(dataset.variables["ap"]) #65
    #print(dataset.variables["b"]) #65
    #print(dataset.variables["p0"])
    #print(dataset.variables["ap0"])  # [  50.  100.  150.  200.  250.  300.  400.  500.  700.  800.  850.  925. 1000.]
    #print(dataset.variables["b0"])  # [0]

    #print(dataset.variables["hybrid0"])  # [0.99555218 0.99851963],  atmosphere_hybrid_sigma_pressure_coordinate
    ##p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)  where ps = surfaceaurpressure p0, ap=a0, b = b0
    #print(dataset.variables["hybrid1"])  # [0.99851963] #
    ##formula: p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    ##formula_terms: ap: ap1 b: b1 ps: surface_air_pressure p0: p01
    #print(dataset.variables["hybrid2"])  # 65 values
    ##formula: p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    ##formula_terms: ap: ap2 b: b2 ps: surface_air_pressure p0: p02


    #print(dataset.variables["height2"])  # [ 80. 120.] height above ground
    #print(dataset.variables["height3"])  # [10.] m  height above ground
    #print(dataset.variables["height_above_msl"])  # [0.] m

    #print(dataset.variables["grib1_vLevel7"][:])  # 0
    #print(dataset.variables["height0"][:])  # 0 m
    #print(dataset.variables["height1"][:])  # 2 m

    #print(dataset.variables["atmosphere_as_single_layer"][:]) #0
    #print(dataset.variables["grib1_vLevel192"][:]) #0
    #print(dataset.variables["grib1_vLevel4"][:]) #0

    #print(np.where(p_pl[:] == 850)[0])
info()

#air_temperature_0m:
#_FillValue: 9.96921E36
#long_name: Surface temperature (T0M)
#standard_name: air_temperature
#units: K
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#_ChunkSizes: 1, 1, 949, 739


#air_temperature_ml:
#_FillValue: -2147483647
#long_name: Air temperature model levels
#standard_name: air_temperature
#units: K
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#scale_factor: 0.001
#_ChunkSizes: 1, 2, 949, 739


# air_temperature_z: Grid
#_FillValue: 9.96921E36
#long_name: Air temperature height levels
#standard_name: air_temperature
#units: K
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#_ChunkSizes: 1, 2, 949, 739



# air_temperature_2m: Grid
#_FillValue: 9.96921E36
#long_name: Screen level temperature (T2M)
#standard_name: air_temperature
#units: K
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#_ChunkSizes: 1, 1, 949, 739



#air_temperature_pl
#_FillValue: -2147483647
#long_name: Air temperature pressure levels
#standard_name: air_temperature
#units: K
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#scale_factor: 0.001
#_ChunkSizes: 1, 1, 949, 739

#surface_air_pressure
#_FillValue: 9.96921E36
#long_name: Surface air pressure
#standard_name: surface_air_pressure
#units: Pa
#grid_mapping: projection_lambert
#coordinates: longitude latitude
#_ChunkSizes: 1, 1, 949, 739
