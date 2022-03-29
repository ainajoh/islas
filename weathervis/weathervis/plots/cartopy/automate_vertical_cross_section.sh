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
if [[ "$HOSTNAME" == *"islas-plotting.novalocal"* ]]; then
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
    modelrun_date=("${1#*=}")
    fi
    ;;
    --modelrun_hour)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
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
modelrun=("${modelrun_date}${modelrun_hour}")
echo $modelrun


#modelrun=("2020022712" "2020022812" "2020022912" "2020030112" "2020030212" "2020030312" "2020030412" "2020030512" "2020030612" "2020030712" "2020030812" "2020030912" "2020031012" "2020031112" "2020031212" "2020031312" "2020031412" "2020031516" "2020031612")
#modelrun=("2020031512")
#modelrun=("2020101012")
id=$$
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do
    # run the vertical cross-sections

    runstring_Vcross="python Vertical_cross_section.py --datetime ${modelrun[i]} --steps ${steps[0]} ${steps[1]} --model $md --id $id --m_level 20 64 --points_lonlat 4.41 20.18 70.55 67.50 --domain_lonlat 3.5 22.0 66.0 72.0 --start_name SEA --end_name KRN --orient 0"
    echo $runstring_Vcross
    $runstring_Vcross
    runstring_Vcross="python Vertical_cross_section.py --datetime ${modelrun[i]} --steps ${steps[0]} ${steps[1]} --model $md --id $id --m_level 20 64 --points_lonlat 20.18 28.27 67.50 67.10 --domain_lonlat 18.0 30.0 65.0 69.0 --start_name KRN --end_name RUS --orient 0"
    echo $runstring_Vcross
    $runstring_Vcross
    runstring_Vcross="python Vertical_cross_section.py --datetime ${modelrun[i]} --steps ${steps[0]} ${steps[1]} --model $md --id $id --m_level 20 64 --points_lonlat 12.1 20.18 78.93 67.50 --domain_lonlat 11.0 22.0 66.0 80.0 --start_name NYA --end_name KRN --orient 1"
    echo $runstring_Vcross
    $runstring_Vcross
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    rm -rf /home/centos/output/weathervis/${modelrun[i]}-$id

  done
done

# fin

