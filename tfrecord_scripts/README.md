# Writing TFrecords

### 1. Set up repositories

```bash
git clone git@github.com:bayer-science-for-a-better-life/abbert-cerebras.git
cd abbert-cerebras
git fetch && git checkout aarti/bayer_esm_predict

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

**NOTE**: In order to generate TFrecords, first change the output folder location for Tfrecords [create_pretraining_data_wrapper.sh#L16](../../../tfrecord_scripts/create_pretraining_data_wrapper.sh)

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