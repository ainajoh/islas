#Useful function for setup
import platform
import os
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

