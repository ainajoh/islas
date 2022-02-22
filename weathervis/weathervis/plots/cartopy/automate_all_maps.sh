#!/bin/bash
source ~/.bashrc

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
steps_max=(1)
domain_name="None"

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
modelrun=(${modelrun_date}${modelrun_hour})
echo ${modelrun}
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


#modelrun=("2020100300") "2020021712" "2020021812" "2020021912"
#modelrun=( "2020022712" "2020022812" "2020022912")
#modelrun=("2020022012" "2020022112" )
#modelrun=("2020021912" "2020022012" "2020022112" "2020022212" "2020022312" "2020022412" "2020022512" "2020022612")
#modelrun=("2020022712" "2020022812" "2020022912" "2020030112" "2020030212" "2020030312" "2020030412" "2020030512" "2020030612" "2020030712" "2020030812" "2020030912" "2020031012" "2020031112" "2020031212" "2020031312" "2020031412" "2020031516" "2020031612")
#modelrun=("2020031512")
#modelrun=("2020100412")
#modelrun=("2020100412")
#modelrun=("2020100512")
#modelrun=("2020100612")
#modelrun=("2020100712")
#modelrun=("2020100812")
#modelrun=("2020100912")
#modelrun=("2020101012")
#modelrun=("2020101012")
echo ${modelrun[i]}
id=$$
#steps=0
for md in ${model[@]}; do
  echo $md
  for ((i = 0; i < ${#modelrun[@]}; ++i)); do
    runstring_WC="python LWC_IWC.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --m_level 24 64 --id $id"
    runstring_IVT="python IVT_IWV.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --m_level 24 64 --id $id"
    runstring_OLR="python OLR_sat.py --datetime ${modelrun[i]} --steps 0 $steps_max  --model $md --domain_name $domain_name --id $id" runstring_BLH="python BLH.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id" runstring_CAO="python CAO_index.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_CT="python Cloud_base_top.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_dxs="python d-excess.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_LMH="python Low_medium_high_clouds.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_Q="python Q_on_levels.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --p_level 800 850 925 --domain_name $domain_name --id $id"
    runstring_SURF="python Surf_conditions.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_T2M="python T2M.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_T850="python T850_RH.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_Z="python Z500_VEL.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_cloud_level="python Low_medium_high_clouds.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    runstring_windlvl="python Wind_on_levels.py --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --p_level 800 850 925 --id $id"
    runstring_wg="python Wind_gusts.py  --datetime ${modelrun[i]} --steps 0 $steps_max --model $md --domain_name $domain_name --id $id"
    
    echo $runstring_CAO
    $runstring_CAO
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
 
    echo $runstring_BLH
    $runstring_BLH
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
 
    echo $runstring_WC
    $runstring_WC 
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_IVT
    $runstring_IVT
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_OLR
    $runstring_OLR
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
     
    echo $runstring_CT
    $runstring_CT 
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_dxs
    $runstring_dxs
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_LMH
    $runstring_LMH
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    
    echo $runstring_Q
    $runstring_Q
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_SURF
    $runstring_SURF	
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_T2M
    $runstring_T2M
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_Z
    $runstring_Z
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    
    echo $runstring_T850
    $runstring_T850
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    
    echo $runstring_wg
    $runstring_wg
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

    echo $runstring_cloud_level
    $runstring_cloud_level
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}
    
    echo $runstring_windlvl
    $runstring_windlvl
    ./converting.sh /home/centos/output/weathervis/${modelrun[i]}-$id ${modelrun[i]}

  done
done

# fin
