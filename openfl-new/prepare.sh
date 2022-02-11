#!/bin/bash
# set -e
workload=$1
config_file=$2

source ${config_file}

# Args: 1 - ip address, 2 - username, 3 - env. activation command, 4 - command
ssh_exec_in_env() {
    ssh $2@$1 "$3 && $4"
}
# Args: 1 - command
authority_exec() {
    ssh_exec_in_env ${authority_ip} ${authority_user} "${authority_env_activation}" "$1"
}
# Args: 1 - command
director_exec() {
    ssh_exec_in_env ${director_ip} ${director_user} "${director_env_activation}" "$1"
}
# Args: 1 - envoy index, 2 - command
envoy_exec() {
    ssh_exec_in_env ${envoys_ips[$1]} ${envoys_users[$1]} "${envoys_env_activation}" "$2"
}
# Args: 1 - command
researcher_exec() {
    bash -c "$1"
}

# cleanup workspaces
cleanup() {
    rm -rf *token
    authority_exec "rm -rf ${authority_workspace}"
    director_exec "rm -rf ${director_workspace}"
    for (( i=0; i<${#envoys_ips[@]}; i++ ));
    do
        envoy_exec $i "rm -rf ${envoys_workspace}-${i}"
    done
    researcher_exec "rm -rf ${researcher_workspace}"
}
cleanup

# certification authority setup
authority_exec "fx pki install --password lol -p ${authority_workspace} --ca-url ${authority_ip}:${authority_port} <<< y"
authority_exec "fx pki run -p ${authority_workspace} > /dev/null 2> /dev/null < /dev/null" &
sleep 3
# get tokens for director, envoys and researcher
authority_exec "fx pki get-token -n ${director_fqdn} --ca-path ${authority_workspace} --ca-url ${authority_ip}:${authority_port}" | xargs | awk -F ' ' '{ print $2 }' > director_token
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    authority_exec "fx pki get-token -n envoy-${i} --ca-path ${authority_workspace} --ca-url ${authority_ip}:${authority_port}" | xargs | awk -F ' ' '{ print $2 }' > envoy-${i}_token
done
authority_exec "fx pki get-token -n researcher --ca-path ${authority_workspace} --ca-url ${authority_ip}:${authority_port}" | xargs | awk -F ' ' '{ print $2 }' > researcher_token

# create director and enyoys workspaces
director_exec "fx director create-workspace -p ${director_workspace}"
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    envoy_exec $i "fx envoy create-workspace -p ${envoys_workspace}-${i}"
done
researcher_exec "mkdir ${researcher_workspace}"

# certify tokens
director_exec "cd ${director_workspace} && fx pki certify -n ${director_fqdn} -t $(cat director_token)" <<< y
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    envoy_exec $i "cd ${envoys_workspace}-${i} && fx pki certify -n envoy-${i} -t $(cat envoy-${i}_token)" <<< y
done
researcher_exec "cd ${researcher_workspace} && fx pki certify -n researcher -t $(cat researcher_token)" <<< y

# copy configs and descriptors to director and envoys
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    scp workloads/${workload}/envoy/envoy_config.yaml ${envoys_users[$i]}@${envoys_ips[$i]}:$(envoy_exec $i "echo ${envoys_workspace}-${i}")
    scp workloads/${workload}/envoy/*.py ${envoys_users[$i]}@${envoys_ips[$i]}:$(envoy_exec $i "echo ${envoys_workspace}-${i}")
    envoy_exec $i "sed -i 's/__ith_envoy__/${i}/' ${envoys_workspace}-${i}/envoy_config.yaml"
done
director_exec "rm ${director_workspace}/director.yaml"
scp workloads/${workload}/director/director_config.yaml ${director_user}@${director_ip}:$(director_exec "echo ${director_workspace}")/

# start director and envoys
director_exec "cd ${director_workspace} && fx director start -c director_config.yaml -rc cert/root_ca.crt -pk cert/${director_fqdn}.key -oc cert/${director_fqdn}.crt > stdout.log 2> stderr.log < /dev/null" &
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    envoy_exec $i "cd ${envoys_workspace}-${i} && fx envoy start -n envoy-${i} --envoy-config-path envoy_config.yaml -dh ${director_fqdn} -dp ${director_port} -rc cert/root_ca.crt -pk cert/envoy-${i}.key -oc cert/envoy-${i}.crt > stdout.log 2> stderr.log < /dev/null" &
done

# start researcher notebook
cp workloads/${workload}/*.py ${researcher_workspace}
cp workloads/${workload}/*.ipynb ${researcher_workspace}
cd ${researcher_workspace}
jupyter notebook
cd ..

# stop director and envoys
director_exec "kill \$(pidof ssh | awk -F ' ' '{ print \$1 }') \$(pidof \$(which python))"
for (( i=0; i<${#envoys_ips[@]}; i++ ));
do
    envoy_exec $i "kill \$(pidof ssh | awk -F ' ' '{ print \$1 }') \$(pidof \$(which python))"
done
# stop authority
authority_exec "kill \$(pidof ssh | awk -F ' ' '{ print \$1 }') \$(pidof \$(which python)) \$(pidof step-ca)"

cleanup
