# How to use config files

## Training configs

Research configs live in `config/train` (all trained on MS1M with `ProxyFreeTrainer`):

| Config | Paper name | Anchoring form | Backbone |
|:---|:---|:---|:---|
| `ms1m_iresnet100_ripple.yml` | RipPLE | arithmetic-mean ENC (`RipPLE` head) | IResNet-100 |
| `ms1m_iresnet100_ripple_r.yml` | SimPLE+ENC | arithmetic-mean ENC + asymmetric temperature `r` (`RipPLE_R` head) | IResNet-100 |
| `ms1m_iresnet100_ripple_gm.yml` | RipPLE-GM | geometric-mean ENC, `explike` norm map (`RipPLE_GM` head) | IResNet-100 |
| `ms1m_iresnet100_ripple_p.yml` | RipPLE-P | per-pair ENC (`RipPLE_P` head) | IResNet-100 |
| `ms1m_iresnet100_simple.yml` | SimPLE | — (baseline) | IResNet-100 |
| `ms1m_sfnet64_ripple.yml` / `ms1m_sfnet64_ripple_r.yml` | RipPLE / SimPLE+ENC | arithmetic-mean ENC | SFNet-64 |

Supervised proxy baselines inherited from OpenSphere also live in `config/train/` (`ms1m_iresnet100_{arcface,cosface,sphereface,sphereface2}.yml`).

For example, to train RipPLE with IResNet-100 on MS1M with 2 GPUs:

```console
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_train.py \
    --cfg_path config/train/ms1m_iresnet100_ripple.yml \
    --proj_dir project/ripple
```

Use `--nproc_per_node=N` (and `CUDA_VISIBLE_DEVICES`) to match your GPU count.

## Testing configs

After training, the model is saved in `project/<method>/<timestamp>`. Test configs live in `config/test`:

- `combined.yml` — LFW, AgeDB-30, CALFW, CPLFW, CFP-FF/FP (pair verification)
- `ijbb.yml` / `ijbc.yml` / `ijb.yml` — IJB-B / IJB-C (template-based 1:1 and 1:N)
- individual validation sets (`lfw.yml`, `agedb.yml`, `calfw.yml`, `cplfw.yml`, `cfp_ff.yml`, `cfp_fp.yml`)

For example, to test a trained model (saved in `project/ripple/20260813_102607`) on IJB-C:

```console
CUDA_VISIBLE_DEVICES=0,1 python main_test.py --cfg_path config/test/ijbc.yml --proj_dir project/ripple/20260813_102607
```

## Customizing configs

See the examples in `config/train` and `config/test`. Key head hyperparameters of the RipPLE family:

- `wrap_mode`: `polarlike` (softplus norm × unit direction) or `explike` (exp norm; required for the exact geometric-mean form)
- `L` / `lambda_`: target norm `ρ` and weight `λ` of the expected norm constraint `loss_L`
- `m`: score margin, an absolute offset on the `ρ`-scale — it moves together with `ρ` in the hyperparameter search (search convention: `m = ρ² × 0.1` for `polarlike`, `m = (ln ρ)² × 0.1` for `explike`)
- `r`: asymmetric temperature (SimPLE+ENC keeps SimPLE's `r`; `r=1` in RipPLE)
- `alpha`: genuine/impostor pair balance weight `α`
