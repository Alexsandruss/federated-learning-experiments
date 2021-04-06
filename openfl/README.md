# openfl Example

Example is executed from $HOME directory
Collaborator and aggregator are physically separate machines accessible through ssh
1. Prepare conda environment:

```
conda create --copy -p ./openfl_env -c conda-forge python=3.8 conda-pack
conda activate ./openfl_env
pip install openfl
```

2. Pack environment and sent it to collaborator/aggregator:
```
# pack
conda-pack -p ./openfl_env -o ./
# unpack
mkdir -p openfl_env
tar -xzf openfl_env.tgz -C openfl_env
source openfl_env/bin/activate
conda-unpack
```

3. Create openfl_workspace on aggregator and collaborator, fill export field in prepare_template.sh and start preparing on aggregator:
```
mkdir -p openfl_workspace && cd openfl_workspace
./prepare_template.sh
# type 'y' and '1' on openfl questions
```

4. Start aggregator and collaborator:
```
# aggregator
fx aggregator start
# collaborator
fx collaborator start -n 1
```
