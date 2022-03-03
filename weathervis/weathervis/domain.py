
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from weathervis.calculation import *

#Preset domain.
if __name__ == "__main__":
    print("Run by itself")

def lonlat2idx(lonlat, url):
    #Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    dataset = Dataset(url)
    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    if len(lonlat)>2:
        idx = np.where((lat > lonlat[2]) & (lat < lonlat[3]) & \
                       (lon >= lonlat[0]) & (lon <= lonlat[1]))
    else:
        print("nearest")
        idx = nearest_neighbour_idx(lonlat[0],lonlat[1],lon,lat)
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
    def __init__(self, date, model, file, lonlat=None, idx=None,domain_name=None, point_name=None, use_latest=True):
        self.date = date
        self.model = model
        self.lonlat = lonlat
        self.idx = idx
        self.domain_name = domain_name
        self.point_name=point_name
        self.use_latest = use_latest
        if type(file)==pd.core.frame.DataFrame:
            self.file = file.loc[0,'File']
        else:
            self.file = file.loc['File']

        YYYY = self.date[0:4]
        MM = self.date[4:6]
        DD = self.date[6:8]
        HH = self.date[8:10]


        if model == "AromeArctic":
            if self.use_latest==False:
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{self.file}?latitude,longitude"
            else:
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/{self.file}?latitude,longitude"

        elif model == "MEPS":
            if self.use_latest==False:
              url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{self.file}?latitude,longitude"
            else:
              url = f"https://thredds.met.no/thredds/dodsC/mepslatest/{self.file}?latitude,longitude"

        self.url = url

        if self.lonlat and not self.idx:
            self.idx = lonlat2idx(self.lonlat, self.url)
        print("IN DOMAIN")
        print(self.point_name)
        print(self.domain_name)
        if self.point_name != None and self.domain_name == None:
            print("GOTCHA")
            sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
            plon = float(sites.loc[self.point_name].lon)
            plat = float(sites.loc[self.point_name].lat)
            self.lonlat = [plon,plat]
            print(self.lonlat)
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
        #eval()

    def MEPS(self):
        self.lonlat = [-1, 60., 49., 72]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Finse(self):
        self.lonlat = [7.524026, 8.524026, 60, 61.5]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def South_Norway(self):
        self.lonlat = [4., 9.18, 58.01, 62.2]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.url)

    def West_Norway(self):
        # self.lonlat = [2., 12., 53., 64.]  # lonmin,lonmax,latmin,latmax,
        self.lonlat = [1.0, 12., 54.5, 64.]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.url)

    def AromeArctic(self):
        # self.lonlat = [-10,60,30,90] #lonmin,lonmax,latmin,latmax,
        self.lonlat = [-18.0, 80.0, 62.0, 88.0]  # [-30,90,10,91] #lonmin,lonmax,latmin,latmax,

        # url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.idx = lonlat2idx(self.lonlat,
                              url=self.url)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def Svalbard_z2(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [15, 23, 77, 82]  #
        self.idx = lonlat2idx(self.lonlat,url)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def Svalbard_z1(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [4, 23, 76.3, 82]  #
        self.idx = lonlat2idx(self.lonlat,url)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def Svalbard(self):  # data
        # url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Svalbard"
        self.lonlat = [-8, 30, 73, 82]  #
        self.idx = lonlat2idx(self.lonlat,url=self.url)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def North_Norway(self):  # data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "North_Norway"
        self.lonlat = [5, 25, 66.5, 77.2]  #
        self.idx = lonlat2idx(self.lonlat,url)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin

    def KingsBay(self):  # bigger data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [10, 13.3, 78.6, 79.3]
        self.idx = lonlat2idx(self.lonlat, url)  # Rough

    def KingsBay_Z0(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [11, 13., 78.73, 79.16]
        self.idx = lonlat2idx(self.lonlat, url)  # Rough

    def KingsBay_Z1(self):  # smaller data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.idx = np.array([[517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
                              520, 520, 520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
                              522, 522, 522, 522, 523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                             [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
                              186, 187, 188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
                              188, 189, 190, 191, 185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188,
                              189]])  # y,x
        self.lonlat = idx2lonlat(self.idx, url)  # rough

    def Andenes(self):
        # 16.120;69.310;10
        self.domain_name = "Andenes"
        self.lonlat = [15.8, 16.4, 69.2, 69.4]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def ALOMAR(self):
        # 16.120;69.310;10
        self.domain_name = "ALOMAR"
        self.lonlat = [15.8, 16.4, 69.2, 69.4]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Andenes_area(self):
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Andenes_area"
        self.lonlat = [6.0, 20.5, 67.5, 71.6]
        #self.lonlat = [12.0, 19.5, 68.0, 70.6]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Varlegenhuken(self):
        point_name = "Varlegenhuken"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Longyearbyen(self):
        point_name = "Longyearbyen"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Hopen(self):
        point_name = "Hopen"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Bodo(self):
        point_name = "Bodo"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Tromso(self):
        point_name = "Tromso"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Bjornoya(self):
        point_name = "Bjornoya"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def NyAlesund(self):
        point_name = "NyAlesund"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Zeppelin(self):
        point_name = "Zeppelin"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Bergen(self):
        point_name = "Bergen"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Kirkenes(self):
        point_name = "Kirkenes"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Kiruna(self):
        point_name = "Kiruna"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Alta(self):
        point_name = "Alta"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Trondheim(self):
        point_name = "Trondheim"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Osteroy(self):
        point_name = "Osteroy"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 1.70)
        maxlon = float(plon + 1.10)
        minlat = float(plat - 0.80)
        maxlat = float(plat + 1.00)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Olsnesnipa(self):  # PAraglidingstart
        point_name = "Olsnesnipa"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def JanMayen(self):  # PAraglidingstart
        point_name = "JanMayen"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def CAO(self):  # PAraglidingstart
        point_name = "JanMayen"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def NorwegianSea(self):  # PAraglidingstart
        point_name = "NorwegianSea"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def NorwegianSea_area(self):  # PAraglidingstart
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "NorwegianSea_area"
        self.lonlat = [-3, 21, 69.0, 77.2]  #
        self.idx = lonlat2idx(self.lonlat, self.url)

    def GEOF322(self):  # PAraglidingstart
        point_name = "GEOF322"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)

    def Iceland(self):
        #url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Iceland"
        #self.lonlat = [12.0, 19.5, 68.0, 70.6]
        #self.idx = lonlat2idx(self.lonlat, self.url)
        #self.lonlat = [-65, 20., 58., 85]
        self.lonlat = [-26., -8, 63., 67]

        self.idx = lonlat2idx(self.lonlat, self.url)

    def pcmet1(self):
        point_name = "pcmet1"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)
    def pcmet2(self):
        point_name = "pcmet2"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)
    def pcmet3(self):
        point_name = "pcmet3"
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.15)
        maxlon = float(plon + 0.15)
        minlat = float(plat - 0.05)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.url)
    def cross_region(self):
        self.domain_name = "cross_region"
        self.idx = []
        self.lonlat = idx2lonlat[self.idx,self.url]
