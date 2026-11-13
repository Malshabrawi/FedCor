# FedCor
The official implementation of the Paper: *FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning* (CVPR 22').

**Abstract:** Client-wise data heterogeneity is one of the major issues that hinder effective training in federated learning (FL). Since the data distribution on each client may vary dramatically, the client selection strategy can significantly influence the convergence rate of the FL process. Active client selection strategies are popularly proposed in recent studies. However, they neglect the loss correlations between the clients and achieve only marginal improvement compared to the uniform selection strategy. In this work, we propose FedCor---an FL framework built on a correlation-based client selection strategy, to boost the convergence rate of FL. Specifically, we first model the loss correlations between the clients with a Gaussian Process (GP). Based on the GP model, we derive a client selection strategy with a significant reduction of expected global loss in each round. Besides, we develop an efficient GP training method with a low communication overhead in the FL scenario by utilizing the covariance stationarity. Our experimental results show that compared to the state-of-the-art method, FedCor can improve the convergence rates by 34% ~ 99% and 26% ~ 51% on FMNIST and CIFAR-10, respectively.


## Required Environment

1. python 3.8
2. pytorch 1.7.0
6. cvxopt 1.2.0

## Running Experiments

The commands in the FMNIST and CIFAR10 sections below include the original
FedCor experiment settings. For the current matched evaluation, use the
500-round configurations in **Matched evaluations and results** below; do not
mix the original CIFAR settings with that evaluation.

| Dataset | Runner dataset name | Model | Model arguments |
| --- | --- | --- | --- |
| Fashion-MNIST | `fmnist` | MLP, hidden layers 64 and 30 | `--dataset=fmnist --model=mlp --mlp_layers 64 30` |
| CIFAR-10 | `cifar10` | CNN, convolution channels 32/64/64 and FC hidden layer 64 | `--dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layers 64` |

The runner translates `--dataset cifar10` to the training argument `--dataset=cifar`.

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



#### Oort

Uses the Fashion-MNIST MLP and matched 500-round training settings. Oort uses one
random cold-start round followed by its own selector, with fixed local epochs.

```shell
python3 src/federated_main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layers 64 30 --epochs=500 --num_users=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=0.005 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --seed 1 2 3 4 5 --oort --run_root=runs/oort_fmnist_dir
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

#### Oort

Uses the CIFAR-10 CNN from the dataset/model table and matched 500-round settings.

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --model=cnn --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layers 64 --epochs=500 --num_users=100 --frac=0.05 --alpha=0.2 --local_ep=3 --local_bs=64 --lr=0.005 --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --seed 1 2 3 4 5 --oort --run_root=runs/oort_cifar10_dir
```

Remove `--gpu=0` for CPU execution. For these direct Oort commands, replace
`--alpha=0.2` with `--shards_per_client=1 --frac=0.1` for SPC1, or
`--shards_per_client=2 --frac=0.05` for SPC2. Do not combine `--oort` with `--gpr`.

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

## Matched evaluations and results

Run from the repository root with the project Python environment active.
`experiments/core.json` defines 500 rounds, 100 clients, three local epochs,
batch size 64, learning rate 0.005 (decay at rounds 150 and 300), seeds 1–5,
and DIR/SPC1/SPC2. GP methods have exactly 15 warm-up rounds.

These commands include all four ablations, Oort/AFL/random comparisons and the
fixed-epoch Full control. Each dataset produces 120 runs. Without `--execute`,
the runner only prints commands. It stops if a training run fails.

```shell
python3 experiments/run.py --dataset fmnist --variants baseline dynamic system full oort afl random full_fixed_epochs --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset cifar10 --variants baseline dynamic system full oort afl random full_fixed_epochs --run_root=runs/main_cifar10 --execute
```

Append `--gpu=0 --gpr_gpu=0` to either main runner command for CUDA. To restrict
a run, add e.g. `--partitions spc1 --seeds 1`; keep such checks in a separate
output root. Do not repeat completed methods into the same root: analysis rejects
duplicate method/seed runs.

Sensitivity evaluates W={5,10,20}, rho={0.1,0.2,0.4}, and penalty={0,0.5,1,2}
one factor at a time (eight unique settings, 240 runs per dataset):

```shell
python3 experiments/sensitivity.py --dataset fmnist --execute
python3 experiments/sensitivity.py --dataset cifar10 --execute
```

The sensitivity runner currently runs on CPU and accepts dataset, partitions,
seeds and execute options only. Each setting has a separate root under
`runs/sensitivity/<runner-dataset>/<parameter-setting>/`.

Each training run has a uniquely named subdirectory containing:

| File | Contents |
| --- | --- |
| `config.json` | Method/component flags, seed, model, settings, package versions and source/partition/profile hashes |
| `partition.json` | Exact ordered sample indices per client |
| `system_profile.json` | Shared simulated communication and computation costs |
| `rounds.jsonl` | Accuracy/loss, selected clients, actual epochs/steps, client validation and participation, thresholds and timing components per round |
| `summary.json` | Completion status, final accuracy, total hybrid time and measured host execution time |
| `legacy_<seed>.pkl` | Legacy training-record format |

A missing `summary.json` indicates a run did not finish successfully. The timing
comparison is simulated parallel client time plus measured server overhead,
including warm-up and GP work; it is not measured distributed wall-clock time.

After all matched runs finish, generate confidence intervals, success counts,
time-to-target and client fairness summaries. Example for
Fashion-MNIST SPC1 using a 60% target:

```shell
python3 analysis/summarize.py runs/main_fmnist --dataset fmnist --partition spc1 --target 0.6 --output runs/analysis/fmnist_spc1.json
```

Repeat per partition and dataset with a target fixed before examining results.
For CIFAR use `--dataset cifar` in the analysis command, even though the runner
uses `cifar10`. Analyze each sensitivity setting separately by passing its root
to the same script. The JSON reports paired accuracy/fairness intervals and
conditional time intervals only for seeds where both methods reach the target;
non-reaching seeds remain explicit. It does not create paper plots automatically.

Full reruns and manuscript updates remain pending; short execution checks do
not establish superiority or convergence. Training outputs and dataset caches
are ignored by Git and stay on the machine where the experiments run.
