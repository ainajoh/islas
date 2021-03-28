#!/bin/bash
source ~/.bashrc

function converting {
  here=$( pwd )
  cd /home/centos/output/weathervis/$1
  # convert to smaller image size and transfer to web disk
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
    copy="scp -i /home/centos/.ssh/islas-key.pem /home/centos/www/gfx/$1/FLEXPART_AA* 158.39.201.233:/home/centos/www/gfx/$1/"
    echo $copy
    $copy
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

#fclagh=350 #3.5 hour before forecsast is issued

if [ "${BASH_VERSINFO:-0}" -ge 4 ];then
  modeldatehour=$(date -u --date "today - $((350*60/100)) minutes" +'%Y%m%d%H%M')
else
  modeldatehour=$(date -v-$((350*60/100))M -u +%Y%m%d%H%M)
  #date -v-60M -u +%Y%m%d%H%M
fi

#modeldatehour="2021022000"

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
release_name="NYA"
domain_name=("AromeArctic" "North_Norway" "Svalbard" "Andenes_area" NorwegianSea_area)
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
    domain_name=("${1#*=}")
    fi
    ;;
    --release_name)
    if [[ "$1" != *=* ]]; then shift;# Value is next arg if no `=`
    release_name="${1#*=}"
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
#domain_name=($domain_name)
echo domain_name
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


#modelrun=("2020022712" "2020022812" "2020022912" "2020030112" "2020030212" "2020030312" "2020030412" "2020030512" "2020030612" "2020030712" "2020030812" "2020030912" "2020031012" "2020031112" "2020031212" "2020031312" "2020031412" "2020031516" "2020031612")
#modelrun=("2021030400")
#steps=0
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do
	  for dom in ${domain_name[@]}; do
    		runstring_FP="python flexpart_AA.py --datetime ${modelrun[i]} --steps 0 ${steps_max} --model $md --domain_name $dom"
		
    		echo $runstring_FP
    		$runstring_FP
    		converting ${modelrun[i]}
	done

  done
done

# fin
