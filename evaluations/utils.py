"""
The script is adapted from https://github.com/microsoft/unilm/blob/master/e5/utils.py
"""

import torch
import logging
import os
import sys

from typing import Mapping, Dict, List
from torch import Tensor
import numpy as np
from transformers import PreTrainedTokenizerFast, BatchEncoding

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import  Counter

from datasets import load_from_disk, concatenate_datasets
import json
import csv

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from tools import tinyllama
from models import *

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class Pooler:
    def __init__(self, pool_type, include_prompt=False):
        self.pool_type = pool_type
        self.include_prompt = include_prompt or self.pool_type in ("cls", "last")

    def __call__(
        self, 
        last_hidden_states: Tensor,
        attention_mask: Tensor,
        prompt_length: int = None,
    ) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        batch_size = last_hidden_states.shape[0]
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        device = last_hidden_states.device
        
        if not self.include_prompt and prompt_length is not None:
            if left_padding:
                prompt_mask = torch.ones_like(attention_mask)
                range_tensor = torch.arange(attention_mask.size(1), 0, -1, device=device).unsqueeze(0)
                prompt_mask = (range_tensor > (sequence_lengths-prompt_length).unsqueeze(1))
                attention_mask[prompt_mask] = 0
            else:
                attention_mask[:, :prompt_length] = 0
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
            attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            emb = s / d
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        elif self.pool_type == "last":
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                emb = last_hidden[torch.arange(batch_size, device=device), sequence_lengths-1]
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")

        return emb


smote = SMOTE(random_state=42)
class Classifier:
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=100)

    def __call__(self, X, true_labels, diable_smote=False):
        min_samples_per_class = 3
        label_counts = {label: sum(1 for l in true_labels if l == label) for label in set(true_labels)}
        X = [x for x, y in zip(X, true_labels) if label_counts[y] >= min_samples_per_class]
        true_labels = [y for y in true_labels if label_counts[y] >= min_samples_per_class]

        X_train, X_test, y_train, y_test = train_test_split(
            X, true_labels, test_size=0.2, random_state=42, stratify=true_labels
        )
        
        if not diable_smote:
            count = Counter(y_train)
            k_neighbors = min(min(count.values()) - 1, 5)
            smote = SMOTE(sampling_strategy={i: count[i] * 5 for i in count.keys()}, k_neighbors=k_neighbors, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "accurcay": accuracy,
            "precision": precision,
            "recall": recall,
            "weighted_f1": weighted_f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }
        return results

def find_best_accuracy(preds, labels):
    labels = np.array(labels)
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_accuracy = 0

    for threshold in thresholds:
        y_pred = (np.array(preds) >= threshold).astype(int)
        accuracy = accuracy_score(labels, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], max_length: int = 512, is_word_level=False) -> BatchEncoding:
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        is_split_into_words=is_word_level,
        return_tensors='pt'
    )


def get_task_def_by_task_name_and_type(task_name: str, task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification']:
        task_name_to_instruct: Dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    if task_type in ['Domain']:
        task_name_to_instruct: Dict[str, str] = {
            'PubMedQA': None, # 'Given a research question and its corresponding abstract, classify the answer as either yes, no, or maybe',
            'ChemProt': None, # 'Given a chemical biology text, classify the relationship between the entities mentioned',
            'MQP': None, # 'Retrieve semantically similar question that asks the same thing',
            'RCT': None, # 'Given a sentence from a medical research abstract, classify the sentence into one of the following categories: background, objective, methods, results, or conclusion',
            'FPB': None,
            'ConvFinQA': None,
            'NER': None,
            'SCOTUS': None,
            'ToS': None,
        }

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)

def get_task_dataset(task_name, tokenizer=None):
    dataset = None
    if task_name == "PubMedQA":
        label_names = ["yes", "no", "maybe"] 
        label2id = {v: i for i, v in enumerate(label_names)}
        with open("data/ori_pqal.json", "r") as file:
            dataset = json.load(file)
        dataset = [[data["QUESTION"] , "".join(data["CONTEXTS"]), data["LONG_ANSWER"], data["final_decision"]] for data in dataset.values()]
        dataset = {
            "sentence": ["Question:" + data[0] + "Context" + data[1] for data in dataset],
            "label": [label2id[data[-1]] for data in dataset]
        }
    elif task_name == "ChemProt":
        label_names = [
            "INHIBITOR", "INDIRECT-DOWNREGULATOR", "SUBSTRATE", "INDIRECT-UPREGULATOR", "ACTIVATOR", "ANTAGONIST", "PRODUCT-OF", 
            "AGONIST", "DOWNREGULATOR" ,"UPREGULATOR", "SUBSTRATE_PRODUCT-OF", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR"
        ] 
        label2id = {v: i for i, v in enumerate(label_names)}
        sentences = []
        labels = []
        with open("data/chemprot.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                sentences.append(data["text"])
                labels.append(label2id[data['label']])
        dataset = {
            "sentence": sentences,
            "label": labels
        }
    elif task_name == "MQP":
        with open("data/mqp.csv", mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            dataset = list(csv_reader)
        dataset = {
            "sentence_1": [data[1] for data in dataset],
            "sentence_2": [data[2] for data in dataset],
            "label": [float(data[3]) for data in dataset]
        }
    elif task_name == "RCT":
        label_names = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"] 
        label2id = {v: i for i, v in enumerate(label_names)}
        with open("data/pubmed_rct.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()

        sentences = []
        labels = []
        for line in lines:
            line = line.strip()
            if line.startswith('###'):
                continue  # 跳过文章编号行
            if '\t' in line:
                label, sentence = line.split('\t', 1)
                sentences.append(sentence.strip())
                labels.append(label2id[label.strip()])
        dataset = {
            "sentence": sentences[:1000],
            "label": labels[:1000]
        }
    elif task_name == "FPB":
        dataset =  load_from_disk("data/fbp")
    elif task_name == "ConvFinQA":
        dataset = load_from_disk("data/fiqa")
        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['valid']])
        dataset = dataset.rename_column("query", "sentence")
        dataset = dataset.rename_column("gold", "label")
    elif task_name == "NER":
        dataset = load_from_disk("data/semeval2017")
        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
        label_names = ["O", "I", "B"] 
        label2id = {v: i for i, v in enumerate(label_names)}
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["document"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples["doc_bio_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
                
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        dataset = dataset.map(tokenize_and_align_labels, batched=True)
        dataset = dataset.rename_column("document", "sentence")
    elif task_name == "SCOTUS":
        dataset = load_from_disk("data/scotus")
        dataset = dataset.rename_column("text", "sentence")
    elif task_name == "ToS":
        dataset_path = "data/tos"
        dataset = load_from_disk(dataset_path)
        dataset = dataset.select(range(3000))
        dataset = dataset.rename_column("text", "sentence")
    return dataset