
cf = "source ../../data/config/config_cyclone.sh"
$cf
#modelrun=("2018031912" "2018031900" "2018031812" "2018031800" "2018031712" "2018031700")
#
#steps_max=(12 24 36 48 60 66)

modelrun=("2020100212")
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

steps_max=(1)

#steps=0
model=("AromeArctic")
domain_name="West_Norway"
for md in ${model[@]}; do
  echo $md
for ((i=0;i<${#modelrun[@]};++i)); do
    runstring_T="python T850_RH.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    runstring_Z="python Z500_VEL.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md"
    runstring_CAO="python CAO_index.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md"
    runstring_SURF="python Surf_conditions.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    runstring_TOA="python TOA_sat.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]}  --model $md"
    runstring_BLH="python BLH.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md"

    #runstring_T="python T850_RH.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    #runstring_Z="python Z500_VEL.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    #runstring_CAO="python CAO_index.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md"
    #runstring_SURF="python Surf_conditions.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    #runstring_TOA="python TOA_sat.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    #runstring_BLH="python BLH.py --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md --domain_name $domain_name"
    ##runstring_sat="python satlookalike --datetime ${modelrun[i]} --steps 0 ${steps_max[i]} --model $md"

    echo runstring_Z
    $runstring_Z
    echo runstring_TOA
    $runstring_TOA
    echo $runstring_T
    $runstring_T
    echo $runstring_CAO
    $runstring_CAO
    echo $runstring_BLH
    $runstring_BLH
    echo $runstring_SURF
    $runstring_SURF

  done
done