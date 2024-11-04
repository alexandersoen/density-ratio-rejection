#! /bin/bash

numfolds=5

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

folds=($(seq 1 1 $numfolds))

for d in "${datasets[@]}"; do
  for n in "${noises[@]}"; do
    for foldidx in "${folds[@]}"; do
      echo "Running Train Rejectors ${d} ${n} (fold ${foldidx}/${numfolds})"
      ipython main.py -- --noise-rate "${n}" --dataset "${d}" --lr 0.001 --batch-size 256 --alpha-rej 3.0 --lamb 1.0 --num-folds "${numfolds}" --fold-idx "${foldidx}"
    done
  done
done
