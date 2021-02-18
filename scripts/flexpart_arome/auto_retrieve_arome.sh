#!/bin/bash
source ~/.bashrc

dirname=$( pwd )
#set workingpath to where this file is located
cd "$(dirname "$0")"

if [[ "$HOSTNAME" == *"cyclone.hpc.uib.no"* ]]; then
    import importlib
    import sys
    from subprocess import call
    #module load Python/3.7.0-foss-2018b
    #source / Data / gfi / users / local / share / virtualenv / dynpie3 / bin / activate
    dname="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/weathervis/weathervis/"
    cyclone_conf = dname + "/data/config/config_cyclone.sh"
    call(f"source {cyclone_conf}", shell=True)
    MODULE_PATH = "/shared/apps/Python/3.7.0-foss-2018b/lib/python3.7/site-packages/netCDF4/__init__.py"
    MODULE_NAME = "netCDF4"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fi

#fclagh=350 #3.5 hour before forecsast is issued

if [ "${BASH_VERSINFO:-0}" -ge 4 ];then
  modeldatehour=$(date -u --date "today - $((350*60/100)) minutes" +'%Y%m%d%H%M')
else
  modeldatehour=$(date -v-$((350*60/100))M -u +%Y%m%d%H%M)
fi

yy=${modeldatehour:0:4}
mm=${modeldatehour:4:2}
dd=${modeldatehour:6:2}
hh=${modeldatehour:8:2}
yymmdd="${yy}${mm}${dd}"

modelrun_date=$yymmdd
chunks=7
steps_min=0
steps_max=48 #65
while [ $# -gt 0 ]; do
  case "$1" in
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
    --steps_max)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    steps_max=("${1#*=}")
    fi
    ;;
    --chunks)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    chunks=("${1#*=}")
    fi
    ;;
    --steps_min)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    steps_min=("${1#*=}")
    fi
    ;;
  --m_level_min)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    m_level_min=("${1#*=}")
    fi
    ;;
  --m_level_max)
    if [[ "$1" != *=* ]]; then shift;  # Value is next arg if no `=`
    m_level_max=("${1#*=}")
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
#make list of list containing correct step values
chunk_list=()
modelrun=${modelrun_date}${modelrun_hour}
tot_steps=$(($steps_max-$steps_min))
splits=$(($tot_steps/$chunks))
for((i=0;i<=$(($splits));++i)) do
  first=$(($steps_min+$chunks*$i))
  second=$(($first+$chunks))
  limit=$(($steps_max-$first))
  if (($limit <= $chunks)); then
    second=$(($first+$limit))
  fi
  if (($first==$second)); then
    break
  fi
  complete=("$first $second")
  chunk_list[${#chunk_list[@]}]=$complete
done
#chunk_list=("7 14" "14 21" "21 28" "28 35" "35 42" "42 48")


for ch in "${chunk_list[@]}"; do #${StringArray[@]}
  echo $ch
done


for dt in ${modelrun[@]}; do #${StringArray[@]}
  for ch in "${chunk_list[@]}"; do #${StringArray[@]}
    runstring="python retrieve_arome.py --steps $ch --datetime $dt --m_levels $m_level_min $m_level_max"
    echo $runstring
    $runstring
  done
done