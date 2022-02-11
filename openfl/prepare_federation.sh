#!/bin/bash

# collaborator
declare -a collaborator_ips=("192.168.16.1" "192.168.16.2" "192.168.16.3" "192.168.16.4")
declare -a collaborator_users=("user" "user" "user" "user")
n_collaborators=${#collaborator_ips[@]}

# Args:
# 1 - ip address
# 2 - user name
# 3 - command
collaborator_exec ()
{
    ssh $2@$1 "source openfl_env/bin/activate && cd /home/$2/openfl_workspace && $3"
}

# clear previous workspace
rm -rf *zip cert/ code logs/ plan/ *txt save .workspace

# aggregator domain name (should be known by collaborator)
export AFQDN=openfl-agg

# WORKSPACE_PATH = $PWD = .../openfl_workspace
export WORKSPACE_TEMPLATE=torch_cnn_mnist
export WORKSPACE_PATH=$PWD

# initialize workspace
fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
sed -i -e "s/conv1_channels_out=20/conv1_channels_out=8/" code/pt_cnn.py
sed -i -e "s/conv2_channels_out=50/conv2_channels_out=12/" code/pt_cnn.py
sed -i -e "s/fc2_insize=500/fc2_insize=100/" code/pt_cnn.py
sed -i -e "s/collaborator_count : 2/collaborator_count : 4/" plan/plan.yaml
bash setup_scripts/${1}.sh
pip install -r requirements.txt
fx plan initialize -a ${AFQDN}
cp mnist_utils.py code/
# create certificates
fx workspace certify
fx aggregator generate-cert-request --fqdn ${AFQDN}
fx aggregator certify --fqdn ${AFQDN} <<< y

# create export archive
fx workspace export

# workspace and certificates sharing with collaborators
for (( i=1; i<${n_collaborators}+1; i++ ));
do
  ssh ${collaborator_users[$i-1]}@${collaborator_ips[$i-1]} "cd openfl_workspace && rm -rf *zip cert/ code logs/ plan/ *txt save .workspace"
  scp openfl_workspace.zip ${collaborator_users[$i-1]}@${collaborator_ips[$i-1]}:/home/${collaborator_users[$i-1]}/
  ssh ${collaborator_users[$i-1]}@${collaborator_ips[$i-1]} "source openfl_env/bin/activate && fx workspace import --archive openfl_workspace.zip"

  collaborator_exec ${collaborator_ips[$i-1]} ${collaborator_users[$i-1]} "fx collaborator generate-cert-request -n ${i}" <<< $i
  scp ${collaborator_users[$i-1]}@${collaborator_ips[$i-1]}:/home/${collaborator_users[$i-1]}/openfl_workspace/col_${i}_to_agg_cert_request.zip .
  fx collaborator certify --request-pkg col_${i}_to_agg_cert_request.zip <<< y
  scp agg_to_col_${i}_signed_cert.zip ${collaborator_users[$i-1]}@${collaborator_ips[$i-1]}:/home/${collaborator_users[$i-1]}/openfl_workspace
  collaborator_exec ${collaborator_ips[$i-1]} ${collaborator_users[$i-1]} "fx collaborator certify --import agg_to_col_${i}_signed_cert.zip"
done
