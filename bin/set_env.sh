#!/bin/bash
# @Author: Muhammad Umair
# @Date:   2024-05-30 16:23:54
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 16:25:01
# Provide all files in the local folder with run access

PROJECT_PATH="$(dirname $(realpath $(dirname -- $0) ))"
# Add project dir to python path 
export PYTHONPATH=$PROJECT_PATH
chmod +x ${PROJECT_PATH}/bin/local/*.sh

