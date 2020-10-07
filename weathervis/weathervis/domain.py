
import numpy as np
from netCDF4 import Dataset
import pandas as pd
#Preset domain.
if __name__ == "__main__":
    print("Run by itself")

def lonlat2idx(lonlat, url):
    #Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    idx = np.where((lat > lonlat[2]) & (lat < lonlat[3]) & \
                   (lon >= lonlat[0]) & (lon <= lonlat[1]))

    idx = np.where((lat > lonlat[2]) & (lat < lonlat[3]) & \
                   (lon >= lonlat[0]) & (lon <= lonlat[1]))
    dataset.close()  #        self.lonlat = [0,30, 73, 82]  #

    return idx

def idx2lonlat(idx, url):
    #todo: Remember if u do this once, you can just copy paste into function and u will only need this when new domain.
    # Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    lon = lon[idx[0].min():idx[0].max(),idx[1].min(): idx[1].max()]
    lat = lat[idx[0].min():idx[0].max(),idx[1].min(): idx[1].max()]
    dataset.close()
    latlon = [lon.min(), lon.max(),lat.min(), lat.max(), ]

    return latlon


class domain():
    def __init__(self, date, model, file, lonlat=None, idx=None):
        self.date = date
        self.model = model
        self.lonlat = lonlat
        self.idx = idx
        self.domain_name = None
        if type(file)==pd.core.frame.DataFrame:
            self.file = file.loc[0,'File']
        else:
            self.file = file.loc['File']

        YYYY = self.date[0:4]
        MM = self.date[4:6]
        DD = self.date[6:8]
        HH = self.date[8:10]
        if model == "AromeArctic":
            url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{self.file}?latitude,longitude"
        elif model == "MEPS":
            url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{self.file}?latitude,longitude"

        self.url = url

        if self.lonlat and not self.idx:
            self.idx = lonlat2idx(self.lonlat, self.url)
        #if self.idx:
        #    self.lonlat = idx2lonlat(self.idx, url)  # rough

        #url = ""#((YYYY==2018 and MM>=9) or (YYYY>2018)) and not (YYYY>=2020 and MM>=2 and DD>=4)
        #if self.model == "MEPS" and ( (int(YYYY)==2018 and int(MM)<9) or ( int(YYYY)<2018 ) ):
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_mbr0_extracted_backup_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"
        #
        #elif self.model == "MEPS" and ( (int(YYYY)==2018 and int(MM)>=9) or (int(YYYY)>2018 )) and ((int(YYYY)==2020 and int(MM)<=2 and int(DD)<4)):
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_mbr0_extracted_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"
        #else:
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_det_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"




    def MEPS(self):
        self.lonlat = [ -1, 60., 49., 72]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Finse(self):
        self.lonlat = [ 7.524026, 8.524026, 60, 61.5]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def South_Norway(self):
        self.lonlat = [4., 9.18, 58.01, 62.2]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.url)

    def West_Norway(self):
        self.lonlat = [2., 12., 53., 64.]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.url)

    def AromeArctic(self):
        #self.lonlat = [-10,60,30,90] #lonmin,lonmax,latmin,latmax,
        self.lonlat = [-30,90,10,91] #lonmin,lonmax,latmin,latmax,

        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.idx = lonlat2idx(self.lonlat,url) # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def Svalbard_z2(self): #map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [15,23, 77, 82]  #
        self.idx = lonlat2idx(self.lonlat,url)# RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
    
    def Svalbard_z1(self): #map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [4,23, 76.3, 82]  #
        self.idx = lonlat2idx(self.lonlat,url)# RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
    def Svalbard(self): #data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Svalbard"
        self.lonlat = [-8,30, 73, 82]  #
        self.idx = lonlat2idx(self.lonlat,url)# RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def KingsBay(self): #bigger data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [10, 13.3, 78.6, 79.3]
        self.idx = lonlat2idx(self.lonlat,url) #Rough

    def KingsBay_Z0(self): #map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [11, 13., 78.73, 79.16]
        self.idx = lonlat2idx(self.lonlat, url) #Rough

    def KingsBay_Z1(self): #smaller data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.idx = np.array([[517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
                     520, 520,520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
                     522, 522, 522, 522,523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                    [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
                     186, 187,188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
                     188, 189, 190, 191,185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189]]) #y,x
        self.lonlat = idx2lonlat(self.idx, url)  # rough


