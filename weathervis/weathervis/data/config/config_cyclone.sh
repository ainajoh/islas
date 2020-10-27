#Needed for module load to work
get_module="source /etc/profile.d/z00_lmod.sh"
$get_module
#Load some required libraries
py3="module load Python/3.7.0-foss-2018b"
dynlib3="source /Data/gfi/users/local/share/virtualenv/dynpie3/bin/activate"
proj="module load PROJ/5.0.0-foss-2018b" #for cartopy to work.

wait
$py3
wait
$dynlib3
wait
$proj

