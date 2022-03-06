#!/bin/bash
# manually run all maps as in crontab
#steps=120
#steps=66
steps=66
runhour=00
#dat=20210323
dat=20220305
#runhour=06
#runhour=12
#runhour=18
#automate_all_meteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
#automate_all_verticalmeteogram_centos.sh --steps \0 $steps --modelrun_hour $runhour 
#automate_cmet_meteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
#automate_cmet_verticalmeteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
#wait


#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway --modelrun_date $date




#automate_all_meteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
#automate_all_verticalmeteogram_centos.sh --steps 0\ $steps --modelrun_hour $runhour 
automate_vertical_cross_section.sh --steps 0\ $steps --modelrun_hour $runhour 
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
#automate_all_maps.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name MEPS
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Osteroy
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name South_Norway
#automate_all_maps.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name West_Norway
#automate_flexpart.sh --model MEPS --steps_max $steps --modelrun_hour $runhour --domain_name Iceland
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
#automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic
#automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard
#automate_flexpartAA.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway
#automate_flexpart.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area
# also run watersip
#automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name AromeArctic  --release_name AN --modelrun $dat
#automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name North_Norway --release_name AN --modelrun $dat
#automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Andenes_area --release_name AN --modelrun $dat
#automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name Svalbard --release_name AN --modelrun $dat
#automate_watersip.sh --model AromeArctic --steps_max $steps --modelrun_hour $runhour --domain_name NorwegianSea_area --release_name AN --modelrun $dat
# fin
