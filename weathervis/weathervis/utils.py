#Useful function for setup
import platform
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable ##__N
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import pandas as pd
import datetime
from netCDF4 import num2date
from weathervis.checkget_data_handler import *
import matplotlib.offsetbox as offsetbox


def setup_directory( path, folder_name):
    projectpath= path+ folder_name
    if not os.path.exists(projectpath):
        os.makedirs(projectpath)
        print("Directory ", projectpath, " Created ")
    else:
        print("Directory ", projectpath, " already exists")
    return projectpath


def adjustable_colorbar_cax(fig1,ax1):#,data, **kwargs):
      #colorbars do not asjust byitself with set_extent when defining a figure size at the start.
      # if u remove figure_size and it adjust, but labels and text will not be consistent.
      # #Some old solutions dont work, this is the best I could find
      divider = make_axes_locatable(ax1) ##__N
      ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes) ##__N
      fig1.add_axes(ax_cb) ##__N
      #cb= fig1.colorbar(data, **kwargs)
      #cb= fig1.colorbar(CF_BLH, fraction=0.046, pad=0.01,aspect=25,cax=ax_cb, label="Boundary layer thickness [m]", extend="both")
      return ax_cb

def nicegrid(ax, xx = np.arange(-20, 80, 20),yy = np.arange(50, 90, 4), color='gray', alpha=0.5, linestyle='--'):
    gl = ax.gridlines(draw_labels=True, linewidth=1, color=color, alpha=alpha, linestyle=linestyle,zorder=10)
    gl.xlabels_top = False
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.FixedLocator(xx)
    gl.ylocator = mticker.FixedLocator(yy)
    gl.xlabel_style = {'color': color}


def remove_pcolormesh_border(xx,yy,data):
    x, y = np.meshgrid(xx,yy)
    nx, ny = x.shape
    mask = (
            (x[:-1, :-1] > 1e20) |
            (x[1:, :-1] > 1e20) |
            (x[:-1, 1:] > 1e20) |
            (x[1:, 1:] > 1e20) |
            (x[:-1, :-1] > 1e20) |
            (x[1:, :-1] > 1e20) |
            (x[:-1, 1:] > 1e20) |
            (x[1:, 1:] > 1e20)
    )
    data = data[ :nx - 1, :ny - 1].copy()
    data[mask] = np.nan
    # ax.pcolormesh(x, y, data[ :, :])#, cmap=plt.cm.Greys_r)
    return x,y,data


def nice_vprof_colorbar(CF, ax, lvl=None, ticks=None, label=None, highlight_val=None, highlight_linestyle="k--",format='%.1f', extend="both",x0=0.75,y0=0.86,width=0.26,height=0.13):
    #x0, y0, width, height = 0.75, 0.86, 0.26, 0.13
    axins = inset_axes(ax, width='80%', height='23%',
                        bbox_to_anchor=(x0, y0, width, height),  # (x0, y0, width, height)
                        bbox_transform=ax.transAxes,
                        loc="upper center")
    cbar = plt.colorbar(CF, extend=extend, cax=axins, orientation="horizontal", ticks=ticks, format=format)
    ax.add_patch(plt.Rectangle((x0, y0), width, height, fc=[1, 1, 1, 0.7],
                                 transform=ax.transAxes, zorder=1000))

    if ticks is not None:
        cbar.ax.xaxis.set_tick_params(pad=-0.5) #all ticks closer attatched to bar

    if highlight_val is not None:
        #only for equally spaced lvls now. Maybe use diff later to correct for irregularies
        #diff = [t[i + 1] - t[i] for i in range(len(t) - 1)] #list of diff between element
        len = lvl[-1] - lvl[0] #17
        us = highlight_val - lvl[0]
        loc_ticks = us/float(len)  #0.5625, 0,975862068965517

        cbar.ax.plot([loc_ticks]*2, [0, 1], highlight_linestyle) #additional contour on plot

    if label is not None:
        cbar.set_label( label, labelpad=-0.5 )
    return cbar

def add_distance_circle():
    print("test")

def add_point_on_map(ax, lonlat = None, point_name=None, labels=None, colors=None):
    #lonlat = [[10,80],[8,60],...[lon,lat]]
    i=0
    colors= mcolors.TABLEAU_COLORS
    iterator = []
    if point_name !=None:
        sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
        lonlat = [sites.loc[point_name].lon, sites.loc[point_name].lat]
        iterator += lonlat
    if lonlat !=None:
        iterator +=lonlat

    for it in iterator:
        lonlat_p = it
        mainpoint = ax.scatter(it[0], it[1], s=9.0 ** 2, transform=ccrs.PlateCarree(),
                            color=colors[i], zorder=6, linestyle='None', edgecolors="k", linewidths=3)
        i+=1

# for creating nice colormaps from hexcode, see https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def plot_track_on_map(dt,model,tim,gca,ccrs,c1,c2,tt, url = '/Data/gfi/isomet/projects/ISLAS_marvin/Data_210327_0545Z'):
    '''
    # dt    : date of mode initiatioin
    # model : name of mode
    # tim   : current time step of the model
    # gca   : current axes of the figure (plt.gca())
    # ccrs  : from cartopy
    # c1    : color of whole track 
    # c2    : color of points in current forecast step (if matched)
    # tt    : current forecast step
    '''
    # url of file needs to be adjusted

    cr = pd.read_csv(url,sep='\s+',skiprows=1,index_col=False,
                     names=['Flight_Time_s', 'Julian_Day',
                            'GpsLon_deg', 'GpsLat_deg', 'GpsAlt_m',
                            'PresAlt_m', 'WindSpeed_m/s', 'WindDir_deg',
                            'Tamb_K, Rh1_%', 'Rh2_%', 'Pa_Pa', 'PV_mA',
                            'Vb_Volts'])


    # now get latest pair of lon/lat that is not NaN
    cr = cr[(cr['GpsLon_deg'].notnull()) & (cr['GpsLat_deg'].notnull())]
    sc1 = gca.scatter(cr['GpsLon_deg'].values, cr['GpsLat_deg'].values
                      ,transform=ccrs.PlateCarree(),
                      s=30, zorder=7,
                      marker='o', linewidths=0.9,
                      c=c1, alpha=0.8,label='CMET track')

    # highlight if the forcast and the ballon have matching windows 
    # first convert the julian dya to datetime
    gg = []
    for i in cr.Julian_Day.values:
        year,julian = [2021,i+13] # 13 days offset in his file
        gg.append(datetime.datetime(year, 1, 1)+datetime.timedelta(days=julian -1))

    # get datetime from model time
    tt = num2date(tt,units='seconds since 1970-1-1 0:0:0')
    # check if day, and hour are agreeing. That should be enough for the short
    # forecast
    idx = [i for i,x in enumerate(gg) if x.day == tt.day and x.hour == tt.hour]
    # when there are matches, plot it 
    if idx:
        lo = cr['GpsLon_deg'].values
        la = cr['GpsLat_deg'].values
        sc2 = gca.scatter(lo[idx],
                          la[idx],
                          transform=ccrs.PlateCarree(),
                          s=30, zorder=7,
                          marker='o', linewidths=0.9,
                          c=c2, alpha=0.8,label='CMET track')
        # create a text field with information about the most recent point
        #nx = nearest_neighbour_idx(lo[idx][-1],la[idx][-1], lat, lon, nmin=1)
        # get the data to write on text field
        para = ['atmosphere_boundary_layer_thickness', 'air_temperature_ml',
                'specific_humidity_ml','surface_air_pressure']
        dmet,data_domain,bad_param = checkget_data_handler(all_param=para, date=dt, model=model, step=tim,
                                                           point_lonlat = [lo[idx][-1], la[idx][-1]])


        # AINA, AINA, HERE, HERE careful! this is height above ground.
        # so only suitable for flight paths over water
        H = [24122.6894480669, 20139.2203688489,17982.7817599549, 16441.7123200128,
         15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
         11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
         9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
         7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
         6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
         4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
         3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
         2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,
         1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
         1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
         952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
         630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
         396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
         230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
         108.885366240509, 83.2097562375566, 58.7032686584901, 34.9801888163106,
         11.6284723290378]
        # this is a bit unnessecary ..
        hy = np.array([0.00986923, 0.02960875, 0.04935756, 0.06913441, 0.08896443,
                       0.10887626, 0.12890122, 0.14907283, 0.16942653, 0.18999945,
                       0.21086545, 0.23210683, 0.25372884, 0.27568749, 0.29793887,
                       0.32043892, 0.34314363, 0.36600904, 0.38899121, 0.41204613,
                       0.43512974, 0.45819812, 0.48120728, 0.50411325, 0.52687202,
                       0.54943958, 0.57177198, 0.59382524, 0.61555536, 0.63691831,
                       0.65787015, 0.67836687, 0.69836451, 0.71781907, 0.73668655,
                       0.75492299, 0.77248438, 0.78932675, 0.80540612, 0.82067847,
                       0.83509984, 0.84865469, 0.86138022, 0.87332184, 0.88450883,
                       0.8949708 , 0.90473767, 0.91383966, 0.92230735, 0.9301717 ,
                       0.93746409, 0.94421627, 0.95046053, 0.95622966, 0.96155705,
                       0.9664767 , 0.97102334, 0.9752326 , 0.97914101, 0.98278628,
                       0.98620762, 0.98944595, 0.99254486, 0.99555218, 0.99851963])
        # get closest level to balloon altitude
        hB   = cr['GpsAlt_m'].values
        absh = np.abs(H -hB[idx][-1])
        hi   = np.where(absh==np.min(absh))[0][0]
        #print(dmet.air_temperature_ml.shape)
        # get relative humdity
        rh = relative_humidity(dmet.air_temperature_ml[0,:,0,0],
                               dmet.specific_humidity_ml[0,:,0,0],
                               dmet.surface_air_pressure[0]*hy)
        #print(rh.shape)
        # now write the textbox
        # Generate text to write.
        text2 = 'BLH: ' + str(int(dmet.atmosphere_boundary_layer_thickness[0]))+' m'
        text3 = 'T: ' + str(int(dmet.air_temperature_ml[0,hi,0,0])) + ' K'
        text4 = 'RH: ' + str(int(rh[0,0,hi])) + '%'
        text = text2 +'\n' + text3 + '\n' + text4
        
        ob = offsetbox.AnchoredText(text, loc=1,
                                    prop=dict(color='black', size=13))
        ob.patch.set(boxstyle='square', color='lightgrey', alpha=0.8)
        gca.add_artist(ob)

    return sc1


def domain_input_handler(dt, model, domain_name, domain_lonlat, file, point_name, point_lonlat=None):
  #print(point_name)
  #print(domain_name)
  #print(domain_lonlat)
  if domain_name or domain_lonlat:
    if domain_lonlat:
      print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
      data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
    else:
      data_domain = domain(dt, model, file=file)

    if domain_name != None and domain_name in dir(data_domain):
      print(f"\n####### Setting up domain: {domain_name} ##########")
      domain_name = domain_name.strip()
      #data_domain = domain(dt, model, file=file, domain_name=domain_name)
      if re.search("\(\)$", domain_name):
        func = f"data_domain.{domain_name}"
      else:
        func = f"data_domain.{domain_name}()"
      print(func)
      print(domain_name)
      eval(func)
    else:
      print(f"No domain found with that name; {domain_name}")
  else:
    data_domain=None
  if (point_name !=None and domain_name == None and domain_lonlat == None):
     print("GGGGGOOOO")
     data_domain = domain(dt, model, file=file, point_name=point_name)
     print("DOM DONE")
  if (point_lonlat != None and point_name == None and domain_name == None and domain_lonlat == None):
     print("IN WRONG ONE")
     data_domain = domain(dt, model, file=file, lonlat=point_lonlat)
     print("DOM DONE")
  print(data_domain)
  return data_domain
