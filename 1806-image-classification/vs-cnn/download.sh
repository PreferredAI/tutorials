#!/bin/bash

# Data
if [ -d "data" ]; then
  echo "'data' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'data.tar.gz' ]; then
    echo 'Data downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/data.tar.gz' -o data.tar.gz
  fi

  echo "Data extracting ..."
  tar -zxf data.tar.gz
  rm data.tar.gz
fi

echo

# Checkpoints
if [ -d "checkpoints" ]; then
  echo "'checkpoints' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'checkpoints.tar.gz' ]; then
    echo 'Checkpoints downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/checkpoints.tar.gz' -o checkpoints.tar.gz
  fi

  echo "Checkpoints extracting ..."
  tar -zxf checkpoints.tar.gz
  rm checkpoints.tar.gz
fi

echo

# Weights
if [ -d "weights" ]; then
  echo "'weights' folder already exists!"
  echo "You need to remove it before downloading again."
else
  if [ ! -f 'weights.tar.gz' ]; then
    echo 'Weights downloading ...'
    curl -L 'https://static.preferred.ai/tutorial/vs-cnn/weights.tar.gz' -o weights.tar.gz
  fi

  echo "Weights extracting ..."
  tar -zxf weights.tar.gz
  rm weights.tar.gz
fi