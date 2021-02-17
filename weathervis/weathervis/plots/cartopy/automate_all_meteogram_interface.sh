
modelrun=("2020100312")
modelrun=("2020100412" "2020100512" "2020100612" "2020100712" "2020100812" "2020100912" "2020101012" "2020101112" "2020101212" "2020101312" "2020101412")

point_num=1
steps=("0" "24")
model="MEPS"
point_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso" "Bjornoya" "NYAlesund" "LONGERYBYEN" "Janmayen" "Norwegiansea" "CAO")
domain_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso" "Bjornoya" "NYAlesund" "LONGERYBYEN" "Janmayen" "Norwegiansea" "CAO")
point_name=("GEOF322")
domain_name=("GEOF322")
#Make overview map
map_loc="python point_maploc.py --datetime ${modelrun[0]} --point_name ${point_name[@]}"

#echo $map_loc
$map_loc

for dt in ${modelrun[@]}; do #${StringArray[@]}
  #runstring="python point_meteogram.py --datetime $dt --point_num $point_num --steps ${steps[0]} ${steps[1]} --model $model"
  runstring="python all_meteogram_interface.py --datetime $dt --point_num $point_num --steps ${steps[0]} ${steps[1]} --model $model"
  for ((i=0;i<${#point_name[@]};++i)); do
    runstring="$runstring --domain_name ${domain_name[i]} --point_name ${point_name[i]}"
    echo $runstring
    $runstring
  done
done
#example of commande:
#python meteogram_v3.py --datetime 2018031700 --point_num 1 --steps 0 60 --model AromeArctic --domain_name Andenes --point_name Andenes


