#!/bin/bash

#Needed for module load to work
get_module="source /etc/profile.d/z00_lmod.sh"
echo "$get_module"
$get_module
#Load some required libraries
py3="module load Python/3.7.0-foss-2018b"
dynlib3="source /Data/gfi/users/local/share/virtualenv/dynpie3-2021a/bin/activate"
proj="module load PROJ/5.0.0-foss-2018b" #for cartopy to work.
wpath="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/weathervis/"
wpath2="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/scripts/flexpart_arome/"

weathervispath="export PYTHONPATH=$PYTHONPATH:$wpath:$wpath2"
wait
echo "$py3"
$py3
wait
echo "$dynlib3"
$dynlib3
wait
echo "$proj"
$proj
wait
echo "$weathervispath"
$weathervispath

