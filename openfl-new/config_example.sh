#!/bin/bash

authority_ip=localhost
authority_port=1234
authority_user=user
authority_workspace=\$HOME/openfl-ca
authority_env_activation="source miniconda3/bin/activate && conda activate openfl-env"

director_ip=localhost
director_port=1235
director_user=user
director_fqdn=openfl-director
director_workspace=\$HOME/openfl-ws-director
director_env_activation="source miniconda3/bin/activate && conda activate openfl-env"

declare -a envoys_ips=(localhost localhost)
declare -a envoys_users=(user user)
envoys_workspace=\$HOME/openfl-ws-envoy
envoys_env_activation="source openfl-env/bin/activate"

researcher_workspace=$PWD/openfl-ws-researcher
