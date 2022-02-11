#!/bin/bash

sed -i -e "s/batch_size         : 256/batch_size         : 64/" plan/plan.yaml
