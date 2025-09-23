# CrossT-GAN

**CrossT-GAN: Large Language Model Empowered Dynamic Adversarial Cross-Domain Time Series Anomaly Detection**

This repository provides the implementation of CrossT-GAN, a novel framework for cross-domain time series anomaly detection. CrossT-GAN dynamically integrates large language modeling (LLM) and adversarial learning for robust generalization across domains with different data distributions.

## 📁 Dataset

- **SMAP**, **MSL**, and **SMD** datasets were obtained from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).
- **SWaT** dataset was obtained from the [TranAD repository](https://github.com/imperial-qore/TranAD).

Before running the code, please make sure the datasets are organized as expected under the corresponding directory (e.g., `./dataset/`). For proprietary reasons, the dataset loader component is not included in the public release. In order to maintain the integrity of the interface, the relevant code interface is still retained, and users can implement the loading code themselves based on the public data and interface specifications.

---

## 🚀 Usage

```bash
## Run Single-Domain Time Series Anomaly Detection

python main_crosstgan_anomaly.py \
    --data_prefix <dataset_name> \
    --strategy linear \
    --adv_rate 0.001 \
    --gpu 0

Arguments:
--data_prefix: specify the dataset name, e.g., SMD, SWaT, SMAP, MSL.
--strategy: weighting strategy for multi-loss fusion, common choices: linear, mlp, etc.
--adv_rate: adversarial loss coefficient, e.g., 0.001.
--gpu: GPU ID to use (default: 0).


## Run a Specific Cross-Domain Combination for Time Series Data Anomaly Detection

python main_crosstgan_domain.py \
    --train_datasets <Multiple_dataset_names> \
    --test_dataset <dataset_name> \
    --strategy linear \
    --adv_rate 0.001 \
    --gpu 0 \
    --latent_dim 128

Arguments:
--train_datasets: comma-separated names of source domain datasets, e.g., "SWaT, SMAP, SMD".
--test_dataset: target domain dataset name, e.g., MSL.
--latent_dim: latent space dimension for domain alignment, e.g., 128.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
