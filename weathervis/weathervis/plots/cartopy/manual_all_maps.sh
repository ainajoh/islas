#!/bin/bash
# manually run all maps as in crontab
#steps=114
steps=66
runhour=06
#runhour=06
#runhour=12
#runhour=18
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name MEPS
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Osteroy
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name South_Norway
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name West_Norway
#automate_flexpart.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Iceland
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
#automate_all_meteogram_centos.sh --steps_max $steps --modelrun_hour $runhour &
#automate_all_verticalmeteogram_centos.sh --steps_max $steps --modelrun_hour $runhour
#wait
# fin
