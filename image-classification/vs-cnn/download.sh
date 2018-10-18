#!/bin/bash

# Data
if [ -d "data" ]; then
  echo "'data' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'data.zip' ]; then
    echo 'Data downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/data.zip' -o data.zip
  fi

  echo "Data extracting ..."
  unzip -q data.zip
  rm data.zip
fi

echo

# Checkpoints
if [ -d "checkpoints" ]; then
  echo "'checkpoints' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'checkpoints.zip' ]; then
    echo 'Checkpoints downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/checkpoints.zip' -o checkpoints.zip
  fi

  echo "Checkpoints extracting ..."
  unzip -q checkpoints.zip
  rm checkpoints.zip
fi

echo

# Weights
if [ -d "weights" ]; then
  echo "'weights' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'weights.zip' ]; then
    echo 'Weights downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/weights.zip' -o weights.zip
  fi

  echo "Weights extracting ..."
  unzip -q weights.zip
  rm weights.zip
fi