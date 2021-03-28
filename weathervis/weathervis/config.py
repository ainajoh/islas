import platform
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(dname)
print(abspath)
#global projectpath
global OUTPUTPATH
OUTPUTPATH = dname + "/../../../../../output/weathervis/"


#package_path = os.path.dirname(__file__)
#os.chdir(dname)
def setup_directory_config(OUTPUTPATH):

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
        print("Directory ", OUTPUTPATH, " Created ")
    else:
        print("Directory ", OUTPUTPATH, " already exists")

    return OUTPUTPATH



def cyclone():
    import importlib
    import sys
    from subprocess import call
    #module load Python/3.7.0-foss-2018b
    #source / Data / gfi / users / local / share / virtualenv / dynpie3 / bin / activate
    cyclone_conf = dname + "/data/config/config_cyclone.sh"
    call(f"source {cyclone_conf}", shell=True)
    MODULE_PATH = "/shared/apps/Python/3.7.0-foss-2018b/lib/python3.7/site-packages/netCDF4/__init__.py"
    MODULE_NAME = "netCDF4"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    OUTPUTPATH = dname+"/../../../../../output/weathervis/"
    OUTPUTPATH = setup_directory(OUTPUTPATH)
    return OUTPUTPATH

def islas_server():
    import importlib
    import sys
    from subprocess import call
    cyclone_conf = dname + "/data/config/config_islas_server.sh"
    call(f"source {cyclone_conf}", shell=True)
    OUTPUTPATH = dname+"/../../../../output/weathervis/"
    OUTPUTPATH = setup_directory(OUTPUTPATH)
    print(OUTPUTPATH)
    return OUTPUTPATH


print("configure")
if "cyclone.hpc.uib.no" in platform.node():
    print("detected cyclone")
    OUTPUTPATH = cyclone()
elif "islas-forecast.novalocal" in platform.node():
    print("detect islas-forecast.novalocal")
    OUTPUTPATH = islas_server()

else:
    print("local host detected")


