"""
The script is adapted from https://github.com/microsoft/unilm/blob/master/e5/mteb_beir_eval.py
"""

import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval, DRESModel
from peft import PeftModel

from utils import *
from task_config import *
from dataclasses import dataclass, field
from transformers import HfArgumentParser

class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, model_args, backward_args=None, **kwargs):
        self.config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **kwargs
        )
        self.config.update(backward_args.to_dict())
        self.encoder = AutoModel.from_pretrained(model_args.model_name_or_path, config=self.config, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, config=self.config, **kwargs)
        self.pooler = Pooler(model_args.pool_type)
        self.keyword = 'sentence_embeddings' if self.encoder.config.model_type == 'nvembed' else 'last_hidden_state'

        if hasattr(self.encoder, "model_init"):
            self.encoder.model_init()
        if backward_args.peft_file is not None:
            model = PeftModel.from_pretrained(
                self.encoder,
                model_args.peft_file,
            )
            self.encoder = model.merge_and_unload()

        self.prefix_type = model_args.prefix_type
        self.doc_as_query = model_args.doc_as_query
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
            prompt_length = 0
        else:
            input_texts = [self.prompt + q for q in queries]
            prompt_length = len(self.tokenizer.encode(self.prompt.strip()))

        return self._do_encode(input_texts, prompt_length)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if self.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if self.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], prompt_length=None) -> np.ndarray:
        encoded_embeds = []
        batch_size = 8 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = self.pooler(outputs[self.keyword], batch_dict['attention_mask'], prompt_length)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


@dataclass
class TaskArgument:
    model_name_or_path: str = field(default="outputs/base/tinyllama-1.1B", metadata={"help": "which model to use"})
    tasks: List[str] = field(default_factory=list, metadata={"help": "a list of MTEB tasks for evaluation"})
    output_dir: str = field(default="outputs/mteb", metadata={"help": "output directory"})
    doc_as_query: bool = field(default=False, metadata={"help": "use query prefix for passages, only used for Quora as it is a symmetric task"})
    pool_type: str = field(default="avg", metadata={"help": "pool type"})
    prefix_type: str = field(default="query_or_passage", metadata={"help": "prefix type"})
    dry_run: bool = field(default=False, metadata={"help": "whether to run the script in dry run mode"})

    def __post_init__(self):
        assert self.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
        assert self.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'

def main():
    parser = HfArgumentParser((TaskArgument, BackwardSupportedArguments))
    args, backward_args = parser.parse_args_into_dataclasses()

    logger.info('Task Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    os.makedirs(args.output_dir, exist_ok=True)

    assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)

    model_kwargs={"torch_dtype": torch.bfloat16,"gguf_file": backward_args.gguf_file, "trust_remote_code": False}

    # if any(keyword in args.model_name_or_path.lower() for keyword in DECODER_MODEL_TYPES):
    #     model_kwargs["attn_implementation"] = "flash_attention_2"

    model = RetrievalModel(args, backward_args, **model_kwargs)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"
    # if model.encoder.config.model_type == "mistral":
    #     model.tokenizer.padding_side = "left"
    print(model.encoder.config)

    args.tasks = [item for task in args.tasks for item in ([task] if task in TASK_LIST else globals()[task])]

    task_names = [t.description["name"] for t in MTEB(
            task_types=['Retrieval'] if not args.tasks else None, 
            tasks=args.tasks or None, 
            task_langs=['en']).tasks
    ]
    task_names = [t for t in task_names if t in TASK_LIST_RETRIEVAL]
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:
        if args.dry_run and task not in ['SciFact', 'FiQA2018']:
            continue

        logger.info('Processing task: {}'.format(task))

        if args.prefix_type == 'query_or_passage':
            args.doc_as_query = task in ['QuoraRetrieval']
        else:
            task_def: str = get_task_def_by_task_name_and_type(task_name=task, task_type='Retrieval')
            prompt: str = get_detailed_instruct(task_def)
            model.set_prompt(prompt=prompt)
            logger.info('Set prompt: {}'.format(prompt))

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(
            model,
            verbosity=1,
            eval_splits=["test" if task not in ['MSMARCO'] else 'dev'],
            output_folder=args.output_dir,
            overwrite_results=False,
            batch_size=1 if task == "ArguAna" else 2,
        )


if __name__ == '__main__':
    main()
