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

lambs="1.0"
alphas="3.0 4.0"
folds=($(seq 1 1 $numfolds))

for lamb in ${lambs[@]}; do
  for d in ${datasets[@]}; do
    for n in ${noises[@]}; do
      foldidx=1
      for foldidx in ${folds[@]}; do
        echo Running Train Rejectors $d $n \( l=$lamb \) \(fold $foldidx/$numfolds\)
        ipython main.py -- --noise-rate $n --dataset $d --lr 0.001 --batch-size 256 --alpha-rej $alphas --num-folds $numfolds --fold-idx $foldidx --lamb $lamb
      done
    done
  done
done
