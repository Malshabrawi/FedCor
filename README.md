# FedCor
The official implementation of the Paper: *FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning* (CVPR 22').

**Abstract:** Client-wise data heterogeneity is one of the major issues that hinder effective training in federated learning (FL). Since the data distribution on each client may vary dramatically, the client selection strategy can significantly influence the convergence rate of the FL process. Active client selection strategies are popularly proposed in recent studies. However, they neglect the loss correlations between the clients and achieve only marginal improvement compared to the uniform selection strategy. In this work, we propose FedCor---an FL framework built on a correlation-based client selection strategy, to boost the convergence rate of FL. Specifically, we first model the loss correlations between the clients with a Gaussian Process (GP). Based on the GP model, we derive a client selection strategy with a significant reduction of expected global loss in each round. Besides, we develop an efficient GP training method with a low communication overhead in the FL scenario by utilizing the covariance stationarity. Our experimental results show that compared to the state-of-the-art method, FedCor can improve the convergence rates by 34% ~ 99% and 26% ~ 51% on FMNIST and CIFAR-10, respectively.


## Required Environment

1. python 3.8
2. pytorch 1.7.0
6. cvxopt 1.2.0

## Running Experiments

Device selection defaults to visible CUDA GPU 0 when available, otherwise CPU.
GP work follows the training device by default. Override with `--gpu=cpu` to
force CPU, `--gpu=1` to choose another visible CUDA device, or `--gpr_gpu=cpu`
/ `--gpr_gpu=1` for separate GP placement. Explicit unavailable GPU requests
fail with a clear error. The same defaults and overrides apply to the main and
sensitivity experiment runners. Non-CUDA accelerator backends use CPU.
Existing example commands with explicit GPU flags override automatic selection.

Cd to the root directory of the repository and run the following command lines to start a training on Dir setting with different client selection strategies. If you want to train with 2SPC or 1SPC setting, you can replace the option ```--alpha=0.2``` with ```--shards_per_client=2``` or ```--shards_per_client=1``` respectively.

For more details about each option, see ```/src/options.py```.

### FMNIST

#### Rand

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5
```

#### FedCor

```shell
python3 src/federated_main.py --gpu=0 --gpr_gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --gpr --poly_norm=0 --GPR_interval=10 --group_size=100 --GPR_gamma=0.95 --update_mean --warmup=15 --discount=0.95
```

#### Pow-d

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --power_d --d=10
```

#### AFL

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 --afl
```



### CIFAR10

#### Rand

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5
```

#### FedCor

```shell
python3 src/federated_main.py --gpu=0 --gpr_gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --gpr --discount=0.9 --GPR_interval=50 --group_size=500 --GPR_gamma=0.99 --poly_norm=0 --update_mean --warmup=20
```

#### Pow-d

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --power_d --d=10
```

#### AFL

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 --epochs=2000 --num_user=100 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 2 3 4 5 --afl
```

### Update
#### 02/10/2023
1. Refine codes and improve the efficiency and stability when training and utilizing GP.
2. Support GP on GPU.
3. Enable updates of the mean in GP.
## FedDynaCor ablation switches

`--system_aware` (alias `--sysetm_aware`) defaults to `false`. Include the flag or set it to
`true` to enable both latency-based selection utility and adaptive local
epochs. Simulated latency is still measured for timing comparisons.
`--dynamic_C` defaults to false; include it to enable dynamic GP selection.
Keep `--dynamic_frac=0` for these ablations so a separate client-count
schedule does not confound the comparison.

Add these options to the same dataset and training command:

| Variant | Options |
| --- | --- |
| Both components disabled | `--gpr --system_aware=false --dynamic_frac=0` |
| Dynamic only | `--gpr --dynamic_C --system_aware=false --dynamic_frac=0` |
| System aware only | `--gpr --system_aware=true --dynamic_frac=0` |
| Full FedDynaCor | `--gpr --dynamic_C --system_aware=true --dynamic_frac=0` |

The straggler penalty is now configurable with `--straggler_penalty` (default 1.0).
Zero is available as an explicit ablation. Adaptive local epochs can be independently
disabled with `--adaptive_local_epochs=false` and are capped by `--max_local_epochs`.

## Experiment configurations

Matched dataset, baseline, sensitivity and epoch-control configurations are in
`experiments/`. Full reruns and manuscript updates remain pending; smoke tests
do not establish superiority or convergence.


## Reduced sensitivity experiments

The main experiments still cover both datasets and all comparison methods.
Sensitivity defaults to **Full FedDynaCor only**, excluding the default setting
already covered by the main experiments. Baseline does not use these parameters
and must not be retrained for every setting.

```shell
python3 experiments/sensitivity.py --dataset fmnist --partitions dir spc1 spc2 --seeds 1 2 --execute
```

This launches seven nondefault settings x three partitions x two seeds (1,2) =
**42 additional 500-round runs**. It changes one factor at a time: W=5/20,
rho=0.1/0.4, penalty=0/0.5/2, around W=10/rho=0.2/penalty=1. Full always retains
adaptive local epochs, including when penalty=0. CUDA/CPU selection remains
automatic. Omit `--execute` to inspect commands. To extend sensitivity to CIFAR-10,
use `--dataset cifar10` (another 42 runs). `--include-reference` optionally trains
the default Full setting too if main results are unavailable; it does not train
baseline. The updated `experiments/run_all.sh` runs 240 main comparisons plus
42 Fashion-MNIST sensitivity runs = **282 runs**; CIFAR sensitivity is optional.

The eight settings cover 48 setting/partition/seed results; six default Full
results are reused from the main suite. Main experiments retain five seeds.
Two seeds provide limited precision; treat sensitivity intervals cautiously.

Results are saved under `runs/sensitivity/fmnist/<parameter-setting>/`. Reuse
matched baseline results directly from `runs/main_fmnist`, for example:

```shell
python3 analysis/summarize.py runs/sensitivity/fmnist/dynamic_th_window-5_dynamic_th_alpha-0.2_straggler_penalty-1.0 --baseline-root runs/main_fmnist --seeds 1 2 --dataset fmnist --partition spc1 --target 0.6 --output runs/analysis/sensitivity_W5_spc1.json
```

Repeat per setting and partition with preselected accuracy targets. Default Full
results remain in the main suite. The analysis checks exact seeds, partition and
latency-profile hashes, code, hardware selection and shared training settings;
incompatible main/sensitivity runs cannot be pooled. No logs are copied and no
baseline training is launched by the analysis. Do not rerun already completed
settings into the same root. Parameter sweeps and analysis do not themselves
produce paper figures.


## Weights & Biases

Install and authenticate in the same environment used for training:

```shell
python3 -m pip install wandb
wandb login
bash run.sh
```

Alternatively set `WANDB_API_KEY` in the training terminal or the platform's
secret/environment settings. Never put the key in run.sh or commit it.
`run.sh` enables project `FedDynaCor` and launches the existing full experiment
shell script. Set `WANDB_PROJECT` and optionally `WANDB_ENTITY` to your project
and team before launching. Each experiment/seed gets a distinct W&B run with
configuration, per-round accuracy/loss, client counts, epochs/steps, timing and
client fairness summaries. Exact client arrays remain in local JSON records.

For direct runner use, export `WANDB_PROJECT=FedDynaCor` first; this also enables
logging for sensitivity. `WANDB_MODE=offline bash run.sh` stores offline W&B
runs inside the local experiment directories for later `wandb sync <run-folder>`.
`WANDB_MODE=disabled bash run.sh` disables W&B. The original
`bash experiments/run_all.sh` remains usable without W&B when WANDB_PROJECT is
unset. Local JSON results remain available; no dataset or checkpoint artifacts
are uploaded by this integration. SDK initialization/upload failures are surfaced
rather than silently claiming successful tracking. Hardware testing and small
checks should use separate output directories from paper experiments.
