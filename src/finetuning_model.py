import os
import sys

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Finetuner:
    def __init__(self, train, valid, is_testing=False):
        self.train = train
        self.valid = valid
        self.data_list = [self.train, self.valid]
        self.text_name = config.TEXT_FEATURE_NAME
        self.normalized_text_name = 'normalized_' + self.text_name
        self.label_names = config.LABEL_FEATURE_NAMES
        self.model_path = config.MODEL_PATH
        self.is_testing = is_testing

    # Preprocess the datasets before loading to the HuggingFace DatasetDict object
    def preprocess(self):
        for dataset in self.data_list:
            dataset.drop([self.text_name], axis=1, inplace=True)
            dataset.drop(dataset.loc[dataset['char_length'] < 5].index, inplace=True)
            dataset.reset_index(drop=True, inplace=True)

    # Load the datasets into HuggingFace DatasetDict object
    def load_data(self):
        if self.is_testing:
            self.raw_datasets = DatasetDict(
                {
                    'train': Dataset.from_pandas(self.train[:10]),
                    'valid': Dataset.from_pandas(self.valid[:10]),
                }
            )
        else:
            self.raw_datasets = DatasetDict(
                {
                    'train': Dataset.from_pandas(self.train),
                    'valid': Dataset.from_pandas(self.valid),
                }
            )

        self.id2label = {idx: label for idx, label in enumerate(self.label_names)}
        self.label2id = {label: idx for idx, label in enumerate(self.label_names)}

    # Tokenize the datasets
    def tokenize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        def tokenize_function(example):
            # encoding
            encoding = self.tokenizer(
                example[self.normalized_text_name], truncation=True, max_length=150
            )
            # add labels
            labels_batch = {
                k: example[k] for k in example.keys() if k in self.label_names
            }
            labels_matrix = np.zeros(
                (len(example[self.normalized_text_name]), len(self.label_names))
            )

            for idx, label in enumerate(self.label_names):
                labels_matrix[:, idx] = labels_batch[label]

            encoding['labels'] = labels_matrix.tolist()

            return encoding

        self.tokenized_datasets = self.raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=self.raw_datasets['train'].column_names,
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.tokenized_datasets.set_format('torch')

    # Define metrics for multi-label classification
    def multi_labels_metrics(self, predictions, labels, threshold=0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_preds = np.zeros(probs.shape)
        y_preds[np.where(probs >= threshold)] = 1

        y_true = labels

        f1_micro_average = f1_score(y_true=y_true, y_pred=y_preds, average='micro')
        roc_auc = roc_auc_score(y_true, y_preds, average='micro')
        accuracy = accuracy_score(y_true, y_preds)

        # return as dictionary
        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}

        for i in range(len(self.label_names)):
            y_preds_i = y_preds[:, i]
            y_true_i = y_true[:, i]
            accuracy_i = accuracy_score(y_true_i, y_preds_i)
            metric_name = 'accuracy_' + self.label_names[i]
            metrics[metric_name] = accuracy_i

        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        result = self.multi_labels_metrics(predictions=preds, labels=p.label_ids)

        return result

    # Build the trainer by passing the model, training arguments, and the datasets
    def build_trainer(self):

        if self.is_testing:
            eval_steps = 10
        else:
            eval_steps = 100

        training_args = TrainingArguments(
            'model_checkpoints',
            evaluation_strategy='steps',
            push_to_hub=False,
            learning_rate=1e-05,
            length_column_name='char_length',
            group_by_length=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            weight_decay=0.25,
            warmup_steps=2,
            warmup_ratio=0.2,
            optim='adamw_hf',
            num_train_epochs=8,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            save_strategy='steps',
            metric_for_best_model='accuracy',
            eval_steps=eval_steps,
            save_steps=eval_steps,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            problem_type='multi_label_classification',
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(self.label_names),
            ignore_mismatched_sizes=True,
        )

        self.trainer = Trainer(
            model,
            training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['valid'],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

    # Run all functions above and train the model
    def train_model(self):
        print('START FINE TUNING...\n')
        self.preprocess()
        self.load_data()
        self.tokenize()
        self.build_trainer()
        self.trainer.train()

    # Return final model evaluation on validation dataset
    def evaluate(self):
        predictions = self.trainer.predict(self.tokenized_datasets['valid'])
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions[0]))
        y_preds = np.zeros(probs.shape)
        y_preds[np.where(probs >= 0.5)] = 1

        y_true = predictions.label_ids

        print(accuracy_score(y_true, y_preds))
        print('\n----------GENERAL REPORT----------')
        print('Accuracy score', accuracy_score(y_true, y_preds))
        print(classification_report(y_true, y_preds))

        for i in range(len(self.label_names)):
            y_preds_i = y_preds[:, i]
            y_true_i = y_true[:, i]
            print('\n----------{} REPORT----------'.format(self.label_names[i].upper()))
            print('Accuracy score', accuracy_score(y_true_i, y_preds_i))
            print(classification_report(y_true_i, y_preds_i))
