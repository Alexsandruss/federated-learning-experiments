#!/bin/bash

sed -i -e "s/conv1_channels_out=20/conv1_channels_out=8/" $1/code/pt_cnn.py
sed -i -e "s/conv2_channels_out=50/conv2_channels_out=16/" $1/code/pt_cnn.py
sed -i -e "s/fc2_insize=500/fc2_insize=128/" $1/code/pt_cnn.py
sed -i -e "s/collaborator_count : 2/collaborator_count : $2/" $1/plan/plan.yaml
