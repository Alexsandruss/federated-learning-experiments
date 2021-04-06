#!/bin/bash

# collaborator
export collaborator_ip=
export collaborator_user=

collaborator_exec ()
{
    ssh ${collaborator_user}@${collaborator_ip} "source openfl_env/bin/activate && cd /home/${collaborator_user}/openfl_workspace && $1"
}

# aggregator domain name (should be known by collaborator)
export AFQDN=

# WORKSPACE_PATH = $PWD = .../openfl_workspace
export WORKSPACE_TEMPLATE=torch_cnn_mnist
export WORKSPACE_PATH=$PWD

# initialize workspace
fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
# reduce number of collaborators to 1
sed -i -e "s/collaborator_count: 2/collaborator_count: 1/" plan/plan.yaml
pip install -r requirements.txt
fx plan initialize -a ${AFQDN}
# create certificates
fx workspace certify
fx aggregator generate-cert-request --fqdn ${AFQDN}
fx aggregator certify --fqdn ${AFQDN}

# export workspace to collaborator
fx workspace export
scp openfl_workspace.zip ${collaborator_user}@${collaborator_ip}:/home/${collaborator_user}/
ssh ${collaborator_user}@${collaborator_ip} "source openfl_env/bin/activate && fx workspace import --archive openfl_workspace.zip"

# certificates sharing
collaborator_exec "fx collaborator generate-cert-request -n 1"
scp ${collaborator_user}@${collaborator_ip}:/home/${collaborator_user}/openfl_workspace/col_1_to_agg_cert_request.zip .
fx collaborator certify --request-pkg col_1_to_agg_cert_request.zip
scp agg_to_col_1_signed_cert.zip ${collaborator_user}@${collaborator_ip}:/home/${collaborator_user}/openfl_workspace
collaborator_exec "fx collaborator certify --import agg_to_col_1_signed_cert.zip"
