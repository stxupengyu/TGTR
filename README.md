These are the datasets and code for our paper "Textual Tag Recommendation with Multi-tag Topical Attention".   
# Requirements:  
* python==3.7.10
* pytorch==1.9.0
* numpy==1.20.3
* pandas==1.3.2
* scikit-learn==0.24.2  
* gensim==4.0.1
* nltk==3.7
* tqdm==4.62.1
* transformers==4.19.2

# Datasets  
We released our processed datasets for reproducibility (and to facilitate related researches).  For pre-processing, we merely removed the HTML Links (e.g., picture address, URL address).  

* [Data](https://drive.google.com/drive/folders/1gUj6zjfn7UzLf9_hUtfYWu0RnyZfiX-f?usp=sharing)


Raw XML files of all datasets are officially published and publicly available.  
* [Raw Data](https://archive.org/details/stackexchange)


# Reproducibility:  

Take the physics dataset as an example.  

## Data Path
Please confirm the corresponding configuration file. Make sure the data path parameters (data_path, txt_path) are right in:   
```bash
myConfig.py
```
## TGTR Experiments  
Comfirm the dataset-specific parameters in: 
```bash
config/physics.py
```

Then, train and evaluate as follows:  
```bash
python main.py -dataset physics
```

## BERT-based TGTR Experiments
Comfirm the dataset-specific parameters in: 
```bash
config/physics.py
```
Comfirm the parameters of BERT in: 
```bash
bert/bertConfig.py
```
Then, train and evaluate as follows:  
```bash
python main.py -dataset physics -encoder bert
```
