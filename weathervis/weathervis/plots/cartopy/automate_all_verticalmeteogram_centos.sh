#!/bin/bash
source ~/.bashrc

function converting {
  here=$( pwd )
  # convert to smaller image size and transfer to web disk
  cd /home/centos/output/weathervis/$1
  if ! [ -d /home/centos/www/gfx/$1 ]; then
    mkdir -p /home/centos/www/gfx/$1
  fi
  for f in *.png; do 
    convert -scale 40% $f /home/centos/www/gfx/$1/$f
    \rm $f
  done
  sudo chown -R centos:apache /home/centos/www/gfx/$1  
  # transfer to webserver
  if [[ "$HOSTNAME" == *"islas-operational.novalocal"* ]]; then
    scp -r -i /home/centos/.ssh/islas-key.pem /home/centos/www/gfx/$1 158.39.201.233:/home/centos/www/gfx
  fi
  cd $here
}

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
steps_max=66
domain_name="None"

point_name=("NyAlesund" "pcmet1" "pcmet2" "pcmet3" "Andenes" "CAO" "NorwegianSea" "Bjornoya" "Longyearbyen" "ALOMAR" "Tromso")
steps="None"

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
    --steps)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    steps=("${1#*=}")
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
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      #exit 1
  esac
  shift
done

echo $model
echo $modelrun_date
echo $modelrun_hour
echo $steps_max
echo $domain_name
modelrun=${modelrun_date}${modelrun_hour}
echo $modelrun


#modelrun=("2020022712" "2020022812" "2020022912" "2020030112" "2020030212" "2020030312" "2020030412" "2020030512" "2020030612" "2020030712" "2020030812" "2020030912" "2020031012" "2020031112" "2020031212" "2020031312" "2020031412" "2020031516" "2020031612")
#modelrun=("2020031512")
#modelrun=("2020101012")
#point_name=("pcmet1" "pcemet2" "pcmet3")
#point_name=("NyAlesund")
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do
      for pnam in ${point_name[@]}; do
	 if [[ ${steps} != "None" ]]
	 then
		echo "test"
		echo "${steps[0]}"
		echo "test2"
	 	runstring_PVmet="python point_vertical_metegram.py --datetime ${modelrun[i]} --steps ${steps[0]} ${steps[1]} --model $md --point_name $pnam"
 	 fi
    echo $runstring_PVmet
    $runstring_PVmet
    converting $modelrun
   done
  done
done

