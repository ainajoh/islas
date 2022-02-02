#!/bin/bash
# run maps from crontab

model=$1 # one of AA, MEPS, FP
runhour=$2
steps=$3

fclag=100 #100 = 1.00 hour before forecsast is issue

echo "${BASH_VERSINFO:-0}"
if [ "${BASH_VERSINFO:-0}" -ge 4 ];then
   modeldatehour=$(date -u --date "today - $(($fclag*60/100)) minutes" +'%Y%m%d%H%M')
else
  modeldatehour=$(date -v-$(($fclag*60/100))M -u +%Y%m%d%H%M)
  #date -v-60M -u +%Y%m%d%H%M
fi
echo $modeldatehour
#modeldatehour="2020030900"
yy=${modeldatehour:0:4}
mm=${modeldatehour:4:2}
dd=${modeldatehour:6:2}
hh=${modeldatehour:8:2}
yymmdd="${yy}${mm}${dd}"
modelrun_hour=$runhour
modelrun_date=$yymmdd

case "$model" in
    AA)
	#url="https://thredds.met.no/thredds/dodsC/aromearcticarchive/${yy}/${mm}/${dd}/arome_arctic_full_2_5km_${modelrun_date}T${modelrun_hour}Z.nc.html"
        url="https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_${modelrun_date}T${modelrun_hour}Z.nc.html"
	echo $url
        web_code=$(curl -sL -w "%{http_code}\n" "$url" -o /dev/null)
        echo $web_code
        echo $(($web_code == 200))
        echo "exit 0" > /home/centos/batch/timeoutwrapper_AA_${runhour}.sh
        if [ $web_code != 200 ]; then
           echo "Data not available on web page yet"
           exit 0
         fi
        echo "exit 64" > /home/centos/batch/timeoutwrapper_AA_${runhour}.sh
	
	automate_all_meteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
	automate_all_verticalmeteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
	#wait
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
    ;;
    MEPS)
        #url="https://thredds.met.no/thredds/dodsC/meps25epsarchive/${yy}/${mm}/${dd}/meps_det_2_5km_${modelrun_date}T${modelrun_hour}Z.nc.html"
        #url="https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_${modelrun_date}T${modelrun_hour}Z.nc.html"
        url="https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_${modelrun_date}T${modelrun_hour}Z.ncml.html"
	echo $url
        web_code=$(curl -sL -w "%{http_code}\n" "$url" -o /dev/null)
        echo $web_code
        echo $(($web_code == 200))
        echo "exit 0" > /home/centos/batch/timeoutwrapper_MEPS_${runhour}.sh
        if [ $web_code != 200 ]; then
           echo "Data not available on web page yet"
           exit 0
         fi
        echo "exit 64" > /home/centos/batch/timeoutwrapper_MEPS_${runhour}.sh

	#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Osteroy
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name South_Norway
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name West_Norway
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name MEPS
    ;;
    FP)
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
	automate_flexpart.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Iceland
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
        # also run watersip
	automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic  --release_name AN
	automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway --release_name AN
    ;;
    FPAA)
	automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
	automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
	automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard 
    ;;
    *)
	printf "***************************\n"
	printf "* Error: Invalid argument.*\n"
	printf "***************************\n"
esac

# fin
