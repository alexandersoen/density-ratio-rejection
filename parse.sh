#! /bin/bash

datasets=(
  "har"
  "gas_drift"
  "mnist"
  "organmnist"
  "octmnist"
  "cifar10"
)

noises=(
  0
  0.25
)

for d in "${datasets[@]}"; do
  for n in "${noises[@]}"; do
    ipython parse.py -- --no-test-noise --alpha-rej 3.0 --num-tau-increments 5 --noise-rate "${n}" --dataset "${d}" --lamb 1 --save-path "json_res/${d}_${n}.json"
  done
done
