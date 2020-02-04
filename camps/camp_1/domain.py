
import numpy as np
from netCDF4 import Dataset

#Preset domain.
if __name__ == "__main__":
    print("Run by itself")

def lonlat2idx(lonlat):
    #Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    idx = np.where((lat > lonlat[2]) & (lat < lonlat[3]) & \
                   (lon >= lonlat[0]) & (lon <= lonlat[1]))
    dataset.close()

    return idx

def idx2lonlat(idx):
    #todo: Remember if u do this once, you can just copy paste into function and u will only need this when new domain.
    # Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    lon = lon[idx[0].min():idx[0].max(),idx[1].min(): idx[1].max()]
    lat = lat[idx[0].min():idx[0].max(),idx[1].min(): idx[1].max()]
    dataset.close()
    latlon = [lon.min(), lon.max(),lat.min(), lat.max(), ]

    return latlon


class DOMAIN():
    def __init__(self, lonlat=None, idx=None):
        self.lonlat = lonlat
        self.idx = idx

    def Arome_arctic(self):
        self.lonlat = [-18.0,80.0,62.0,88.0] #lonmin,lonmax,latmin,latmax,

        self.idx = np.array([[0,948],[0,738]]) #Index y,x

    def Svalbard(self):
        self.lonlat = [10,19, 77, 80]  #
        self.idx = lonlat2idx(self.lonlat)# RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def KingsBay_Z0(self):
        self.lonlat = [11, 14,78.8, 79.2]
        self.idx = lonlat2idx(self.lonlat) #Rough

    def KingsBay_Z1(self):
        self.idx = np.array([[517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
                     520, 520,520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
                     522, 522, 522, 522,523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                    [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
                     186, 187,188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
                     188, 189, 190, 191,185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189]]) #y,x
        self.lonlat = idx2lonlat(self.idx)  # rough


