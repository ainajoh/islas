#Useful function for setup
import platform
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable ##__N
import matplotlib.pyplot as plt
import numpy as np

def domain_input_handler(dt, model, domain_name, domain_lonlat, file):

    if domain_name or domain_lonlat:
        if domain_lonlat:
            print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
            data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
        else:
            data_domain = domain(dt, model, file=file)

        if domain_name != None and domain_name in dir(data_domain):
            print(f"\n####### Setting up domain: {domain_name} ##########")
            domain_name = domain_name.strip()
            if re.search("\(\)$", domain_name):
                func = f"data_domain.{domain_name}"
            else:
                func = f"data_domain.{domain_name}()"
            eval(func)
    else:
        data_domain = None
    return data_domain

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
    gl = ax.gridlines(draw_labels=True, linewidth=1, color=color, alpha=alpha, linestyle=linestyle)
    gl.xlabels_top = False
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.FixedLocator(xx)
    gl.ylocator = mticker.FixedLocator(yy)

