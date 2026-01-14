# CrossT-GAN

**CrossT-GAN: Large Language Model Empowered Dynamic Adversarial Cross-Domain Time Series Anomaly Detection**

This repository provides the implementation of CrossT-GAN, a novel framework for cross-domain time series anomaly detection. CrossT-GAN dynamically integrates large language modeling (LLM) and adversarial learning for robust generalization across domains with different data distributions.

## 📁 Dataset

- **SMAP**, **MSL**, and **SMD** datasets were obtained from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).
- **SWaT** dataset was obtained from the [TranAD repository](https://github.com/imperial-qore/TranAD).

Before running the code, please ensure that the datasets are organized under the expected directory structure (e.g., ./dataset/). To accommodate heterogeneous data formats and preprocessing conventions across benchmarks, dataset loaders are designed as pluggable, user-implementable components based on unified abstract interfaces. This design enables flexible integration of diverse datasets while keeping the model implementation dataset-agnostic. Users may implement dataset-specific preprocessing logic by following the provided interface specifications. Detailed preprocessing procedures and experimental settings are fully described in the paper.


---

## 🚀 Usage

```bash

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
