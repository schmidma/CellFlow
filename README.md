# CharmingSyringes

Download the weights from <https://zenodo.org/record/8063977/files/full_run.ckpt?download=1>

Modify `inference.sh` by pointing `--root_dir` to the parent directory of the data (where `a`, `b`, `c`, ... reside)
Point `--from_checkpoint` to the downloaded checkpoint file.
Prediction output directory can be specified via `--pred_dir`.

Inference via

```sh
sbatch inference.sh -r aihero-gpu
```
