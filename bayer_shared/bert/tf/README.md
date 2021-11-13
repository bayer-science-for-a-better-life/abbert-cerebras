# Feature Extraction from Trained models

### 1. Set up repositories

```bash
git clone git@github.com:bayer-science-for-a-better-life/abbert-cerebras.git
cd abbert-cerebras
git fetch && git checkout aarti/feature_extract

# Clone modelzoo
git clone git@github.com:Cerebras/modelzoo.git
```

### 2. Create conda environment for inference on GPU

Please refer to the file [`bayer_conda_env.yml`](../../../bayer_conda_env.yml)

```bash
conda env create -f ./abbert-cerebras/bayer_conda_env.yml
conda activate bayer
pip install -e .
```

### 3. Create TFrecords 
TFRecords should be created using `abbert2/tfrecord_scripts/create_tfrecords.py` file. Note that we need to have a text file with locations of all the parquet files as shown in sample text files provided at `abbert2/tfrecord_scripts/bayer_subsets` folder

**NOTE**: In order to generate TFrecords, first change the output folder location for Tfrecords [create_pretraining_data_wrapper.sh#L17](../../../tfrecord_scripts/create_pretraining_data_wrapper.sh)

```bash
cd tfrecord_scripts

source create_parallel_tfrecs.sh <path to text file with folder paths>
```
```
# For example: 
source create_parallel_tfrecs.sh /cb/home/aarti/ws/code/bayer_tfrecs/tfrecord_scripts/bayer_subsets/bayer_dirs_09.txt

```

Sample content of text file to be passed to `create_parallel_tfrecs.sh` is as below
```bash
/cb/customers/bayer/oas-processed/paired/Alsoiussi_2020/SRR11528762_paired
/cb/customers/bayer/oas-processed/paired/Alsoiussi_2020/SRR11528761_paired
/cb/customers/bayer/oas-processed/paired/Eccles_2020/SRR10358525_paired
/cb/customers/bayer/oas-processed/paired/Eccles_2020/SRR10358523_paired
/cb/customers/bayer/oas-processed/paired/Eccles_2020/SRR10358524_paired
/cb/customers/bayer/oas-processed/paired/Goldstein_2019/SRR9179294_paired
.
.
.
.
```

### 4. Modify config yaml files 

After generating TFrecords, we need to modify the config yamls with the correct path 

Please edit the files `predict_input/data_dir` part of config yamls located [here](./configs/feature_extract). A sample config file is provided for your reference - [gpu_params_roberta_base_heavy_sequence_small_dataset_predict.yaml](./configs/gpu_params_roberta_base_heavy_sequence_small_dataset_predict.yaml)

The config files provided are for the various runs that were performed on CS-1

### 5. Download trained model checkpoints
The trained model checkpoint folders are uploaded to the S3 bucket and their corresponding yaml config files are present here [here](./configs/feature_extract)


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









