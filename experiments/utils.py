import os
from collections import Counter
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from tools import tinyllama
from models import *

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png', figure_title=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    mask = np.zeros_like(attention, dtype=bool)
    mask[attention == 0] = True

    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    cmap.set_bad(color='gray')  # 设置掩码部分的颜色为灰色

    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap=cmap,
                     annot=annot,
                     mask=mask,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    # ax.set_yticklabels(x_texts, rotation=0, fontsize=26)
    # ax.tick_params(axis='x', labelsize=32)
    # ax.tick_params(axis='y', labelsize=32, rotation=0)

    if figure_title is not None:
        ax.set_title(figure_title, fontsize=60)
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)

    plt.savefig(os.path.join(figure_path, figure_name), dpi=100)
    plt.close()

def replacef(text):
    if text == '<s>':
        return '[CLS]'
    elif text == '</s>':
        return '[SEP]'
    return text.replace('Ġ', '').replace('▁', '')

LABEL_LIST =  ['O', 'PER', 'PER', 'ORG', 'ORG', 'LOC', 'LOC', 'MISC', 'MISC']
LABEL_DICT = {i: label for i, label in enumerate(LABEL_LIST)}

def visualize(trainer, dataset, output_dir="pics"):
    tem = trainer.model.config.output_attentions
    trainer.model.config.output_attentions = True
    tokenizer=trainer.tokenizer

    sample_num = 16

    random.seed(43)
    sample_ids = random.sample(range(len(dataset)), sample_num)
    samples = dataset.select(sample_ids)
    predictions = trainer.predict(samples.remove_columns("labels")).predictions
    attentions = predictions[-1]
    attentions = [np.stack([attention[id] for attention in attentions]) for id in range(sample_num)]

    for index in [0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        for id, sample, attention in zip(range(sample_num), samples, attentions):
            texts = tokenizer.convert_ids_to_tokens(sample["input_ids"])
            x_texts = [replacef(text) for text in texts]
            y_texts = [text + f"{(LABEL_DICT.get(label, 'O')):>5}" for text, label in zip(x_texts, sample["labels"])]
            if id != 6:
                continue
            seq_length = len(texts)
            attention = attention[index][:, :seq_length, :seq_length]
            attention_mean = np.mean(attention, axis=0)
            attention_plot(attention_mean, annot=True, x_texts=x_texts, y_texts=y_texts, figsize=(15, 15),
                        # figure_path=os.path.join(output_dir, f"layer_{index}"), 
                        figure_path=output_dir, 
                        figure_name=f'attention_weight_{index}.png', figure_title=f"Layer {index+1}")
    trainer.model.config.output_attentions = tem


def cal_corr(trainer, dataset):
    import numpy as np
    from scipy.stats import spearmanr
    tem = trainer.model.config.output_hidden_states
    trainer.model.config.output_hidden_states = True
    sample_num = 4
    compare_first = True

    random.seed(42)
    sample_ids = random.sample(range(len(dataset)), sample_num)
    samples = dataset.select(sample_ids)
    try:
        samples = samples.remove_columns("labels")
    except:
        samples = samples.remove_columns("label")

    predictions = trainer.predict(samples).predictions
    hidden_state = predictions[1][-1]
    attention_mask = samples['attention_mask']

    bsz, seq_len = hidden_state.shape[:2]
    correlations = np.zeros((bsz, seq_len))
    sentence_corr = np.zeros((bsz))

    compare_first = True
    for i, mask in enumerate(attention_mask):
        seq_len = len(mask)
        pivot_token = 0 if compare_first else seq_len - 1
        for j in range(seq_len):
            if j == pivot_token:
                continue
            corr, _ = spearmanr(hidden_state[i, pivot_token, :], hidden_state[i, j, :])
            correlations[i, j] = corr
        sentence_corr[i] = np.sum(correlations[i]) / (seq_len - 1)
    
    avg_corr = np.mean(sentence_corr)

    compare_first = False
    for i, mask in enumerate(attention_mask):
        seq_len = len(mask)
        pivot_token = 0 if compare_first else seq_len - 1
        for j in range(seq_len):
            if j == pivot_token:
                continue
            corr, _ = spearmanr(hidden_state[i, pivot_token, :], hidden_state[i, j, :])
            correlations[i, j] = corr
        sentence_corr[i] = np.sum(correlations[i]) / (seq_len - 1)
    
    avg_corr_2 = np.mean(sentence_corr)
    trainer.model.config.output_hidden_states = tem
    return avg_corr, avg_corr_2

def cal_corr_loop(trainer, dataset):
    import numpy as np
    from scipy.stats import spearmanr
    tem = trainer.model.config.output_hidden_states
    trainer.model.config.output_hidden_states = True
    sample_num = 100
    compare_first = True

    random.seed(42)
    sample_ids = random.sample(range(len(dataset)), sample_num)
    samples = dataset.select(sample_ids)
    try:
        samples = samples.remove_columns("labels")
    except:
        samples = samples.remove_columns("label")

    best_config = set([21])
    best_corr = 0
    for k in range(10):
        best_select = None
        for l in range(22):
            if l in best_config:
                continue
            config = best_config | {l}
            trainer.model.base_model.unsink_layers = config

            predictions = trainer.predict(samples).predictions
            hidden_state = predictions[1][-1]
            attention_mask = samples['attention_mask']

            bsz, seq_len = hidden_state.shape[:2]
            correlations = np.zeros((bsz, seq_len))
            sentence_corr = np.zeros((bsz))

            compare_first = True
            for i, mask in enumerate(attention_mask):
                seq_len = len(mask)
                pivot_token = 0 if compare_first else seq_len - 1
                for j in range(seq_len):
                    if j == pivot_token:
                        continue
                    corr, _ = spearmanr(hidden_state[i, pivot_token, :], hidden_state[i, j, :])
                    correlations[i, j] = corr
                sentence_corr[i] = np.sum(correlations[i]) / (seq_len - 1)
            
            avg_corr = np.mean(sentence_corr)

            compare_first = False
            for i, mask in enumerate(attention_mask):
                seq_len = len(mask)
                pivot_token = 0 if compare_first else seq_len - 1
                for j in range(seq_len):
                    if j == pivot_token:
                        continue
                    corr, _ = spearmanr(hidden_state[i, pivot_token, :], hidden_state[i, j, :])
                    correlations[i, j] = corr
                sentence_corr[i] = np.sum(correlations[i]) / (seq_len - 1)
            
            avg_corr_2 = np.mean(sentence_corr)
            if (avg_corr) > best_corr:
                best_corr = avg_corr
                best_select = l
            print(f"[{k},{l},config{config}] cls : {avg_corr}, last : {avg_corr_2}, best : {best_select}")
        if best_select is None:
            break
        print(f"[{k} epoch]: select {best_select} layer")
        best_config.update({best_select})
    print(f"select {best_config} layer")

    trainer.model.config.output_hidden_states = tem
    return best_config

def statistics(predictions, datasets):
    labels = datasets["labels"]
    tokens = datasets["tokens"]

    count = 0
    for pred, label, token in zip(predictions, labels, tokens):
        valid_label = [(l, p, "T" if l==p else "F") for p, l in zip(pred, label) if l != -100]
        if all([l == p for p, l, t in valid_label]):
            continue
        # if len(valid_label) > 5:
        #     continue
        # df = pd.DataFrame(valid_label, columns=["label", "pred", "correct"])
        valid_token = token[:len(valid_label)]
        data = [[t, *v] for t, v in zip(valid_token, valid_label)]
        df = pd.DataFrame(data, columns=["token", "label", "pred", "correct"])
        
        print(df.T)
        count += 1
        if count > 10:
            break
    print({i:v for i, v in enumerate(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])})

    count_dict = Counter()
    valid_example = 0
    for pred, label in zip(predictions, labels):
        valid_label = [(l, p) for p, l in zip(pred, label) if l != -100]
        valid_length = len(valid_label)
        # if valid_length < 10:
        #    continue
        valid_example += 1
        count_dict.update([valid_length])
            # #int(i / valid_length * 32) 
            # valid_length
            # for i, (l, p) 
            # in enumerate(valid_label) if l != p])
    df = pd.DataFrame([(key, value / valid_example) for key, value in count_dict.items()], columns=['Key', 'Value'])
    sns.barplot(x='Key', y='Value', data=df)
    plt.savefig('counter_data_plot.png', dpi=300, bbox_inches='tight')

def plot_from_json(json_path, keyword="loss", title=None):
    import json
    import matplotlib.pyplot as plt

    title = title if title is not None else json_path

    with open(os.path.join(json_path, "trainer_state.json"), 'r') as f:
        data = json.load(f)

    data = data['log_history']
    
    losses = [log[keyword] for log in data if keyword in log]

    steps = [log['step'] for log in data if keyword in log]

    # 绘制图像
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', markersize=5)
    plt.xlabel('Step')
    plt.ylabel(keyword)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(json_path, f"plot_{keyword}.png"))
