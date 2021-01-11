#Needed for module load to work
get_module="source /etc/profile.d/z00_lmod.sh"
echo "$get_module"
$get_module
#Load some required libraries
py3="module load Python/3.7.0-foss-2018b"
dynlib3="source /Data/gfi/users/local/share/virtualenv/dynpie3-2021a/bin/activate"
proj="module load PROJ/5.0.0-foss-2018b" #for cartopy to work.
wpath="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/weathervis/"
weathervispath="export PYTHONPATH=$PYTHONPATH:$wpath"
wait

$py3
wait
$dynlib3
wait
$proj
wait
$weathervispath

