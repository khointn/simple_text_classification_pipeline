# Simple Pipeline for Fine-tuning Text Classification Task (Draft 1)

## Introduction

An end-to-end pipeline for multi-label text classification. This tool has 3 main functions:
* Preprocesses your text data and fine-tune with pretrained Transformers. 
* Provides API so that you can test your model.

Input:

- Dataset consist of: text column, label column(s).
- Provide text column name and label name(s).
- Provide the url to the HuggingFace model for fine-tuning.

## How to run

1/ Drop your datasets (train, valid) to the `python simple_text_classification_pipeline/data` directory, or you can define path to your datasets later.

2/ Create Conda env

```python
conda create -n text_clf python=3.8.8
```

3/ Install necessary requirements

```python
python -m pip install -r requirements.txt
```

4/ Run command

```python
cd src
python main.py --label_name label1 label2 --text_name text --model_path your/pretrained/path
```

`python label1 label2` is your label names (default label_name = [label1, label2])

`python text` is your text column name (default text_name = text). 

`python your/pretrained/path` is the path to your (HuggingFace) pretrained model (default = microsoft/Multilingual-MiniLM-L12-H384).

<br /><br />
**Use case: If you want to define your data paths**

```python
python main.py /path/to/train /path/to/valid --label_name label1 label2 --text_name text --model_path your/pretrained/path
```

`python /path/to/train`: default = simple_text_classification_pipeline/data/train.csv

`python /path/to/valid`: default = simple_text_classification_pipeline/data/valid.csv

<br /><br />
**Use case: Run a trial with 10 samples before running with full batch**

```python
python main.py --label_name label1 label2 --text_name text --model_path your/pretrained/path --is_testing True
```

`python is_testing`: default = False

<br /><br />
**Use case: Create API after training**

```python
python main.py --label_name label1 label2 --text_name text --model_path your/pretrained/path --return_api True
```

`python return_api`: default = False

## Functions

As we run `python python main.py`, the code execute as follow:

1. Preprocessing: config datatype, normalize texts, drop duplicate, remove non-latin texts, remove stopwords & punctuations.
2. Fine-tuning: config training arguments and start fine-tuning.
3. Return API link (if requested)



