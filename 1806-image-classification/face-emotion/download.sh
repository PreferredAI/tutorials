#!/bin/bash

if [ -d "data" ]; then
  echo "'data' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'data.tar.gz' ]; then
    echo 'Data downloading ...'
    echo
    curl -L 'https://static.preferred.ai/tutorial/face-emotion/data.tar.gz' -o data.tar.gz
  fi

  echo "Data extracting ..."
  tar -zxf data.tar.gz
  rm data.tar.gz
  echo "Data is ready!"
fi