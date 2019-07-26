#!/usr/bin/env bash


module load python/3.6.4

venv -m .env

python -m venv .env

source .env/bin/activate

pip3 install numpy

python3 main.py
