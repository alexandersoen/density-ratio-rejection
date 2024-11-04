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
  if [ "$d" = "har" ] || [ "$d" = "gas_drift" ]; then
    batchsize=64
  else
    batchsize=256
  fi

  epochs=1

  for n in "${noises[@]}"; do
    for foldidx in "${folds[@]}"; do
      echo "Running Train base clf ${d} ${n} (fold ${foldidx}/${numfolds})"
      ipython main_clf.py -- --noise-rate "${n}" --dataset "${d}" --lr 0.001 --batch-size "${batchsize}" --epochs "${epochs}" --num-folds "${numfolds}" --fold-idx "${foldidx}"
    done
  done
done
