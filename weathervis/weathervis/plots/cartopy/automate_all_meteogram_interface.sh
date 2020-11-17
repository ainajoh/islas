#Made 20201117
#Runs all_meteogram_interface.py and point_maploc.py
#These two python scripts runs again: point_maploc.py, point_vertical_meteogram.py and point_meteogram.py
#############################################################################
modelrun=("2018031700")
point_num=1
steps=(0 4)
model="AromeArctic"
#NB: point_name and domain_name has to be equal length, so if u want to run another point in the same domain you dublicate the domain_name
point_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso-Holt" "Bjornoya" "NYAlesund" "MetBergen")
domain_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso-Holt" "Bjornoya" "NYAlesund" "MetBergen")
#Make overview map
map_loc="python point_maploc.py --datetime ${modelrun[0]} --point_name ${point_name[@]}"
echo $map_loc
$map_loc

for dt in $modelrun; do
  runstring="python all_meteogram_interface.py --datetime $dt --point_num $point_num --steps ${steps[0]} ${steps[1]} --model $model "
  for ((i=0;i<${#point_name[@]};++i)); do
    runstring="$runstring --domain_name ${domain_name[i]} --point_name ${point_name[i]}"
    echo $runstring
    $runstring
  done
done
#example of commande:
#python meteogram_v3.py --datetime 2018031700 --point_num 1 --steps 0 60 --model AromeArctic --domain_name Andenes --point_name Andenes


