#!/bin/bash

if [ -d "data" ]; then
  echo "'data' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'data.zip' ]; then
    echo 'Data downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/face-emotion/data.zip' -o data.zip
  fi

  echo "Data extracting ..."
  unzip -q data.zip
  rm data.zip
  echo "Data is ready!"
fi