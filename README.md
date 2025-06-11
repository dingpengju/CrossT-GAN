# CrossT-GAN
CrossT-GAN: Large Language Model Empowered Dynamic Adversarial Cross-Domain Time Series Anomaly Detection
# Dataset
The SMAP dataset, MSL dataset, and SMD dataset were obtained from https://github.com/NetManAIOps/OmniAnomaly. The SWaT dataset was obtained from https://github.com/imperial-qore/TranAD.
# Run single-domain time series data anomaly detection
python main_crosstgan_anomaly.py \
    --data_prefix <dataset_name> \
    --strategy linear \
    --adv_rate 0.001 \
    --gpu 0
    
--data_prefix: specify the dataset name, e.g. SMD, SWaT, SMAP, MSL.
--strategy: weighting strategy, common values are linear, mlp, etc.
--adv_rate: adversarial rate hyperparameter.
--gpu: GPU ID, default is 0.

# Run a specific cross-domain combination for time series data anomaly detection
python main_crosstgan_domain.py \
    --train_datasets "SWaT, SMAP, SMD" \
    --test_dataset "MSL" \
    --strategy linear \
    --adv_rate 0.001 \
    --gpu 0 \
    --latent_dim 128

--train_datasets: comma-separated source domain dataset names.
--test_dataset: target domain dataset.
--latent_dim: potential representation dimension.
