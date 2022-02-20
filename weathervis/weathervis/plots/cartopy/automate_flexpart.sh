#!/bin/bash
source ~/.bashrc

function converting {
  here=$( pwd )
  # convert to smaller image size and transfer to web disk
  cd /home/centos/output/weathervis/$1
  if ! [ -d /home/centos/www/gfx/$1 ]; then
    mkdir /home/centos/www/gfx/$1
  fi
  for f in *.png; do 
    convert -scale 40% $f /home/centos/www/gfx/$1/$f
    \rm $f
  done
  sudo chown -R centos:apache /home/centos/www/gfx/$1  
  # transfer to webserver
  if [[ "$HOSTNAME" == *"islas-operational.novalocal"* ]]; then
    scp -i /home/centos/.ssh/islas-key.pem /home/centos/www/gfx/$1/F* 158.39.201.233:/home/centos/www/gfx/$1
  fi
  cd $here
}

#dirname=$( pwd )
dirname=""
#set workingpath to where this file is located
echo "$(dirname "$0")"
cd "$(dirname "$0")"

#echo $hostname
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

#fclagh=2400 #24 hours before forecsast is issued

if [ "${BASH_VERSINFO:-0}" -ge 4 ];then
  modeldatehour=$(date -u --date "today - $((2400*60/100)) minutes" +'%Y%m%d%H%M')
else
  modeldatehour=$(date -v-$((2400*60/100))M -u +%Y%m%d%H%M)
  #date -v-60M -u +%Y%m%d%H%M
fi

#modeldatehour="2021022000"
#modeldatehour="2021031500"
modeldatehour="2021040500"

yy=${modeldatehour:0:4}
mm=${modeldatehour:4:2}
dd=${modeldatehour:6:2}
hh=${modeldatehour:8:2}
yymmdd="${yy}${mm}${dd}"

#yymmddhh=${yymmdd}${hh}
modelrun_date=$yymmdd
modelrun_hour="00"
model=("AromeArctic")
steps_max=(1)
domain_name="None"
release_name="NYA"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
    if [[ "$1" != *=* ]]; then shift; # Value is next arg if no `=`
    model=("${1#*=}")
    fi
    ;;
    --modelrun)
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
    --steps_max)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    steps_max=("${1#*=}")
    fi
    ;;
    --domain_name)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    domain_name="${1#*=}"
    fi
    ;;
    --release_name)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    release_name="${1#*=}"
    fi
    ;;
    *)
      printf "******************************\n"
      printf "* Error: Invalid argument $1.*\n"
      printf "******************************\n"
      exit 1
  esac
  shift
done

echo $model
echo $modelrun_date
echo $modelrun_hour
echo $steps_max
echo $domain_name
echo $release_name
modelrun=("${modelrun_date}${modelrun_hour}")
echo $modelrun
#modelrun=("2021010100")
#model=("AromeArctic")
#steps_max=(1)
#domain_name="West_Norwa
#domain_name=""
#model=("$1")
#modelrun=("$2")
#steps_max=($3)
#if [$4]
#then
#  domain_name="$4" #West_Norway
#fi

#modelrun=("2021022300")
#modelrun=("2021022500")
#modelrun=("2021022600")

#steps=0
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do

    python flexpart_EC.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]} ${modelrun[i]}
  done
done

# fin
