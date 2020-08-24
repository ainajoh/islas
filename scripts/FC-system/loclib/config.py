import platform

if "cyclone.hpc.uib.no" in platform.node():
    cyclone()

def cyclone():
    import importlib
    import sys
    #module load Python/3.7.0-foss-2018b
    #source / Data / gfi / users / local / share / virtualenv / dynpie3 / bin / activate

    MODULE_PATH = "/shared/apps/Python/3.7.0-foss-2018b/lib/python3.7/site-packages/netCDF4/__init__.py"
    MODULE_NAME = "netCDF4"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)



