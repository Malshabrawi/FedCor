# FedDynaCor smoke tests

Four component configurations completed on Fashion-MNIST with Dirichlet alpha 0.2.
These are execution checks, not accuracy or speedup validation.

## Configuration

Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, NumPy 1.26.4, cvxopt 1.3.3; CPU with one thread.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/federated_main.py --gpr --dataset=fmnist --model=mlp --mlp_layers 16 --num_users=100 --frac=0.05 --epochs=31 --warmup=25 --local_ep=1 --local_bs=256 --iid=0 --alpha=0.2 --seed 1 --lr=0.005 --dynamic_frac=0 --verbose=0 <variant options>
```

The loop uses zero-based epoch > warmup, so 31 rounds give five GP-selection rounds after warm-up 25.

| Variant | Options | Rounds | GP selection rounds | Finite GP diagnostics | Exit code |
| --- | --- | --- | --- | --- | --- |
| baseline | `--system_aware=false` | 31 | 5 | 5 | 0 |
| dynamic | `--dynamic_C --system_aware=false` | 31 | 5 | 5 | 0 |
| system | `--system_aware=true` | 31 | 5 | 5 | 0 |
| full | `--dynamic_C --system_aware=true` | 31 | 5 | 5 | 0 |

## Runtime fixes verified

- Dynamic selection retains at least one client before applying the stopping threshold.
- Empty selections return an empty latency tensor rather than an undefined variable.
- GP posterior diagnostics use a symmetric double-precision linear solve to avoid float32 covariance failures.
- Reporting the cross-entropy scalar detaches it from autograd.

Additional focused checks verified system utility on/off with a nonzero penalty, minimum participation, empty selections, and finite GP predictions across several conditioning sets.

## Limitations

The configured straggler penalty is still zero, so the selection utility multiplier is one even when system awareness is enabled. Adaptive local epochs remain controlled by the system-aware switch. The existing threshold sign convention and latency accounting have not been revised. The dynamic runs selected one client per post-warm-up round with these settings.

An initial 10-client Dirichlet setup failed its allocation solver; the successful runs use the documented 100 clients. Two pre-fix dynamic runs crashed on empty selection; all four final runs above were repeated after the fixes.
