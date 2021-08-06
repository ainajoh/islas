#!/bin/bash
source ~/.bashrc

#set workingpath to where this file is located
echo "$(dirname "$0")"
cd "$(dirname "$0")"

cf=""
if [[ "$HOSTNAME" == *"cyclone.hpc.uib.no"* ]]; then
    cf="source ../../data/config/config_cyclone.sh"
fi
if [[ "$HOSTNAME" == *"islas-forecast.novalocal"* ]]; then
    cf="source ../../data/config/config_islas_server.sh"
fi
if [[ "$HOSTNAME" == *"islas-operational.novalocal"* ]]; then
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


yy=${modeldatehour:0:4}
mm=${modeldatehour:4:2}
dd=${modeldatehour:6:2}
hh=${modeldatehour:8:2}
yymmdd="${yy}${mm}${dd}"

#yymmddhh=${yymmdd}${hh}
modelrun_date=$yymmdd
modelrun_hour="00"
model=("AromeArctic")
steps_max=(66)
domain_name="None"
point_name=("Andenes" "pcmet1" "pcmet2" "pcmet3" "ALOMAR" "Tromso" "NyAlesund" "NorwegianSea" "Bjornoya" "CAO" "Longyearbyen")

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
    if [[ "$1" != *=* ]]; then shift; # Value is next arg if no `=`
    	model=("${1#*=}")
    fi
    ;;
    --modelrun_date)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    	echo "teeees"
    modelrun_date=("${1#*=}")
    fi
    ;;
    --modelrun_hour)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    	echo "teeees"
    modelrun_hour=("${1#*=}")
    fi
    ;;
    --domain_name)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    	domain_name="${1#*=}"
    fi
    ;;
    --point_name)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    	point_name=("${1#*=}")
    fi
    ;;
    --steps)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    	steps=("${1#*=}")
    fi
    ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      #exit 1
  esac
  shift
done
echo $point_name
echo ${point_name}
echo $model
echo $modelrun_date
echo $modelrun_hour
echo $steps_max
echo $domain_name
modelrun=("${modelrun_date}${modelrun_hour}")
echo $modelrun
id=$$

point_name=("Andenes" "pcmet1" "pcmet2" "pcmet3" "ALOMAR" "Tromso" "NyAlesund" "NorwegianSea" "Bjornoya" "CAO" "Longyearbyen")
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do
      for pnam in ${point_name[@]}; do
	  if [[ ${steps} != "None" ]]
          then
	        runstring_Pmet="python point_meteogram.py --datetime ${modelrun[i]} --steps ${steps[0]} ${steps[1]} --model $md --domain_name $pnam --point_name $pnam --id $id"
          fi

    echo $runstring_Pmet
    $runstring_Pmet
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    rm -f /home/centos/output/weathervis/${modelrun[i]}-$id
    done
  done
done

