# Feature Extraction from Trained models

### 1. Set up repositories

```bash
git clone git@github.com:bayer-science-for-a-better-life/abbert-cerebras.git
cd abbert-cerebras
git fetch && git checkout aarti/bayer_esm_predict

# Clone modelzoo
git clone git@github.com:Cerebras/modelzoo.git
```

### 2. Create conda environment for inference on GPU

Please refer to the file [`bayer_conda_env.yml`](../../../cerebras_bayer_conda_env.yml)

```bash
conda env create -f ./abbert-cerebras/bayer_conda_env.yml
conda activate bayer
pip install -e .
```

### 4. Modify config yaml files 

After generating TFrecords, we need to modify the config yamls with the correct path 

Please edit the files `predict_input/data_dir` part of config yamls located [here](./configs/feature_extract).
A sample config file is provided for your reference - [gpu_params_roberta_base_heavy_sequence_small_dataset_predict.yaml](./configs/gpu_params_roberta_base_heavy_sequence_small_dataset_predict.yaml)

The config files provided are for the various runs that were performed on CS-1

### 5. Download trained model checkpoints

The trained model checkpoint folders are uploaded to the S3 bucket and their corresponding yaml config files are present [here](./configs/feature_extract)


### 6. Launch feature extraction

In order to extract features i.e encoder outputs from encoder layers, please use the following command as reference

```bash

cd bayer_shared/bert/tf
conda activate bayer

# Check correct python
which python

python run_feature_extract.py --mode=predict --params=./configs/feature_extract/cs1_params_roberta_base_heavy_sequence_LRSwarm_decay_bsz1k_msl164_cdr_25_25_50.yaml --checkpoint_path=<path/to/trained_model_folder>/cs1_params_roberta_base_heavy_sequence_LRSwarm_decay_bsz1k_msl164_cdr_25_25_50/model.ckpt-540000 --model_dir=<path_to_folder_where_extracted_feature_files_to_be_stored>

```

`model_dir` contains generated parquet files which have the features extracted from all encoder layers.
