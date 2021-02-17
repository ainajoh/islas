modelrun=("2021021300")
steps="0 48"
chunks=( "7 14" "14 21" "21 28" "28 35" "35 42" "42 48")


for dt in ${modelrun[@]}; do #${StringArray[@]}
  for ch in "${chunks[@]}"; do #${StringArray[@]}
    runstring="python retrieve_arome.py --steps ${ch} --datetime $dt"
    echo $runstring
    $runstring
  done
done