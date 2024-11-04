# Code SI

The following is code for ["Rejection via Learning Density
Ratios"](https://openreview.net/forum?id=JzcIKnnOpJ), NeurIPS 2024.

The python requirements can be found in the `environment.yml` file.

## Datasets

All dataset except those provided by `PyTorch` need to be downloaded. We have
provided a script to be run in the root directory: `download_data.sh`.
For manual download, one can check the bash script.

## Running code

To run the code, we have a series of bash scripts to use.
Parameters (and their ranges) can be changed.
The current scripts only present a subset of the ranges used to make the plots.

```bash
bash clf.sh
bash baseline.sh
bash rej.sh
bash parse.sh
```

These, respectively, train the base classifier, train the density ratio
rejectors, and evaluate and parse test set results.

## Citation

```bibtex
@inproceedings{
  soen2024rejection,
  title={Rejection via Learning Density Ratios},
  author={Soen, Alexander and Husain, Hisham and Schulz, Philip and Nguyen, Vu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=JzcIKnnOpJ}
}
```
