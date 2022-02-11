import subprocess as sp
import os
from multiprocessing import Pool
from sys import argv


def cmd(command):
    res = sp.run(command, shell=True, stdout=sp.PIPE,
                         stderr=sp.PIPE, encoding='utf-8')
    return res.stdout[:-1], res.stderr[:-1]

commands = [
    'fx aggregator start',
    'ssh dietpi@192.168.16.1 "source openfl_env/bin/activate && cd openfl_workspace && fx collaborator start -n 1"',
    'ssh dietpi@192.168.16.2 "source openfl_env/bin/activate && cd openfl_workspace && fx collaborator start -n 2"',
    'ssh dietpi@192.168.16.3 "source openfl_env/bin/activate && cd openfl_workspace && fx collaborator start -n 3"',
    'ssh dietpi@192.168.16.4 "source openfl_env/bin/activate && cd openfl_workspace && fx collaborator start -n 4"'
]

os.system(f'mkdir -p {argv[1]}_logs')
with Pool(len(commands)) as p:
    outputs = p.map(cmd, commands)
    for i, output in enumerate(outputs):
        if i == 0:
            with open(f'{argv[1]}_logs/aggregator.log', 'w') as log_file:
                log_file.write(output[0])
        else:
            with open(f'{argv[1]}_logs/collaborator_{i}.log', 'w') as log_file:
                log_file.write(output[0])
