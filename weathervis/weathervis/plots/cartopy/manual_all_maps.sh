#!/bin/bash
# manually run all maps as in crontab
steps=24
#runhour=00
#runhour=06
runhour=12
#runhour=18
automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name MEPS
# fin
