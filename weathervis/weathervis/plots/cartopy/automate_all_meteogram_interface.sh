#!/bin/bash
source ~/.bashrc

cf=""
if [[ "$HOSTNAME" == *"cyclone.hpc.uib.no"* ]]; then
    cf="source ../../data/config/config_cyclone.sh"
    fi
if [[ "$HOSTNAME" == *"islas-forecasts-testing.novalocal"* ]]; then
    cf="source ../../data/config/config_islas_server.sh"
    fi

$cf

#fclagh=350 #3.5 hour before forecsast is issued

if [ "${BASH_VERSINFO:-0}" -ge 4 ];then
  modeldatehour=$(date -u --date "today - $((350*60/100)) minutes" +'%Y%m%d%H%M')
else
  modeldatehour=$(date -v-$((350*60/100))M -u +%Y%m%d%H%M)
  #date -v-60M -u +%Y%m%d%H%M
fi

#modeldatehour="2021021121"

yy=${modeldatehour:0:4}
mm=${modeldatehour:4:2}
dd=${modeldatehour:6:2}
hh=${modeldatehour:8:2}
yymmdd="${yy}${mm}${dd}"

#yymmddhh=${yymmdd}${hh}
modelrun_date=$yymmdd
modelrun_hour="00"
modelrun=( ${modelrun_date}${modelrun_hour} )
echo ${modelrun[0]}

#modelrun=("2020100312")
#modelrun=("2020100412" "2020100512" "2020100612" "2020100712" "2020100812" "2020100912" "2020101012" "2020101112" "2020101212" "2020101312" "2020101412")

point_num=1
steps=("0" "24")
#model="MEPS"
model="AromeArctic"
#point_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso" "Bjornoya" "NYAlesund" "LONGERYBYEN" "Janmayen" "Norwegiansea" "CAO")
#domain_name=("Andenes" "VARLEGENHUKEN" "HOPEN" "BODO" "Tromso" "Bjornoya" "NYAlesund" "LONGERYBYEN" "Janmayen" "Norwegiansea" "CAO")
#point_name=("GEOF322")
#domain_name=("GEOF322")
point_name=("Andenes")
domain_name=("Andenes")
#Make overview map
map_loc="python point_maploc.py --datetime ${modelrun[0]} --point_name ${point_name[@]}"
echo $map_loc
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

# fin
