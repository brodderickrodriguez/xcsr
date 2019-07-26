#!/usr/bin/env bash


module load python/3.6.4

python -m venv .env

source .env/bin/activate

pip3 install numpy
pip3 install matplotlib

python3 main.py
