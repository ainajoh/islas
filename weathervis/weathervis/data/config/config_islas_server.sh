#!/bin/bash

conda_env="conda activate weathervis"
wpath="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/weathervis/"
weathervispath="export PYTHONPATH=$PYTHONPATH:$wpath"
echo "$conda_env"
$conda_env

echo "$weathervispath"
$weathervispath

