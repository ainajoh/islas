#!/bin/bash
# run maps from crontab

model=$1 # one of AA, MEPS, FP
runhour=$2
steps=$3

case "$model" in
    AA)
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
	automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
    ;;
    MEPS)
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Osteroy
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name South_Norway
	automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name West_Norway
	#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name MEPS
    ;;
    FP)
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard --release_name NYA
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard --release_name HH
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic --release_name NYA
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic --release_name HH
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway --release_name NYA
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway --release_name HH
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area --release_name NYA
	automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area --release_name HH
    ;;
    *)
	printf "***************************\n"
	printf "* Error: Invalid argument.*\n"
	printf "***************************\n"
esac

# fin
