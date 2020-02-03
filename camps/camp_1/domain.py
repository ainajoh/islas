
import numpy as np
from netCDF4 import Dataset

#Preset domain.
if __name__ == "__main__":
    print("Run by itself")

def latlon2idx(latlon):
    #Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    idx = np.where((lat > latlon[0]) & (lat < latlon[1]) & \
                   (lon >= latlon[2]) & (lon <= latlon[3]))
    dataset.close()

    return idx

def idx2latlon(idx):
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
    latlon = [lat.min(), lat.max(), lon.min(), lon.max()]

    return latlon


class DOMAIN():
    def __init__(self, latlon=None, idx=None):
        self.latlon = latlon
        self.idx = idx

    def Arome_arctic(self):
        self.latlon = [62.0,88.0,-18.0,80.0] #latmin,latmax,lonmin,lonmax
        self.idx = np.array([[0,-1],[0,-1]]) #Index y,x

    def Svalbard(self):
        self.latlon = [77, 80, 10, 13]  # latmin,latmax,lonmin,lonmax
        self.idx = latlon2idx(self.latlon)# RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def KingsBay_Z0(self):
        self.latlon = [78.8, 79.2, 11, 14]
        self.idx = latlon2idx(self.latlon) #Rough

    def KingsBay_Z1(self):
        self.idx = np.array([[517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
                     520, 520,520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
                     522, 522, 522, 522,523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                    [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
                     186, 187,188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
                     188, 189, 190, 191,185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189]]) #y,x
        #self.idx = np.array([[183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
        #                      186, 187, 188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
        #                      188, 189, 190, 191, 185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188,
        #                      189], [517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
        #                      520, 520, 520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
        #                      522, 522, 522, 522, 523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525]])  # y,x
        self.latlon = idx2latlon(self.idx)  # rough


