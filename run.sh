#!/bin/bash

# This script run a docker container for specified "Algo" on specified "PORT"
function run() {
  id=$1
  port=$2
  name=$(echo $id | sed -e "s/_/-/g")
  printf "import streamlit as st\\nfrom run import run;\\ntry:\\n\\trun()\\nexcept:\\n\\tst.error('Something went wrong!')" > $name.py
  cat Dockerfile > Dockerfile.$id
  sed -i "s/Chapter-ID/$id/g" Dockerfile.$id
  sed -i "s/Chapter-Name/$name/g" Dockerfile.$id

  lower=$(echo "$name" | awk '{print tolower($0)}')
  sudo docker build -f Dockerfile.$id -t $lower .
  sudo docker rm -f $lower
  sudo docker run --name="$lower" -d --restart=unless-stopped -p $port:8501 $lower
  rm $name.py
  rm Dockerfile.$id
}

run "Introduction" 8501
run "Linear_Regression" 8502
run "Logistic_Regression" 8503
run "K_Means_Clustering" 8504
