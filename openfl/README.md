# OpenFL Example

Example is executed from $HOME/openfl_workspace directory with $HOME/openfl_env environment

Collaborators and aggregator are physically separate and ssh accessible machines

1. Prepare conda environment:

```
conda create --copy -p ./openfl_env -c conda-forge python=3.8 conda-pack
conda activate ./openfl_env
pip install openfl # only for x86-64 linux
```

2. Pack environment and sent it to collaborators/aggregator:
```
# pack
conda-pack -p ./openfl_env -o ./
# unpack
mkdir -p openfl_env
tar -xzf openfl_env.tgz -C openfl_env
source openfl_env/bin/activate
conda-unpack
```

3. Create openfl_workspace on aggregator and collaborators, fill export field in prepare_template.sh and start preparing on aggregator:
```
mkdir -p openfl_workspace && cd openfl_workspace
./prepare_template.sh
# type 'y' and collaborator index on openfl questions
```

4. Start aggregator and collaborators:
```
# aggregator
fx aggregator start
# collaborators
fx collaborator start -n 1
fx collaborator start -n 2
```
