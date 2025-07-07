"""
The script is adapted from https://github.com/microsoft/unilm/blob/master/e5/mteb_except_retrieval_eval.py
"""

import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel
from typing import List
from mteb import MTEB

from utils import *
from task_config import *
from dataclasses import dataclass, field
from transformers import HfArgumentParser


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_args, backward_args=None, **kwargs):
        super().__init__()
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

        self.l2_normalize = True
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.eval()
        self.encoder.cuda()

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        prompt_length = len(self.tokenizer.encode(self.prompt.strip()))
        input_texts: List[str] = [self.prompt + s for s in sentences]

        encoded_embeds = []
        batch_size = 8 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = self.pooler(outputs[self.keyword], batch_dict['attention_mask'], prompt_length)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

@dataclass
class TaskArgument:
    model_name_or_path: str = field(default="outputs/base/tinyllama-1.1B", metadata={"help": "which model to use"})
    task_types: List[str] = field(default_factory=list, metadata={"help": "task types to evaluate"})
    tasks: List[str] = field(default_factory=list, metadata={"help": "a list of MTEB tasks for evaluation"})
    output_dir: str = field(default="outputs/mteb", metadata={"help": "output directory"})
    doc_as_query: bool = field(default=False, metadata={"help": "use query prefix for passages, only used for Quora as it is a symmetric task"})
    pool_type: str = field(default="avg", metadata={"help": "pool type"})
    prefix_type: str = field(default="query_or_passage", metadata={"help": "prefix type"})
    multilingual: bool = field(default=False, metadata={"help": "whether to use multilingual model"})
    dry_run: bool = field(default=False, metadata={"help": "whether to run the script in dry run mode"})

    def __post_init__(self):
        assert self.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
        assert self.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'

def main():
    parser = HfArgumentParser((TaskArgument, BackwardSupportedArguments))
    args, backward_args = parser.parse_args_into_dataclasses()

    logger.info('Task Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    os.makedirs(args.output_dir, exist_ok=True)

    model_kwargs={"torch_dtype": torch.bfloat16,"gguf_file": backward_args.gguf_file, "trust_remote_code": False}

    # if any(keyword in args.model_name_or_path.lower() for keyword in DECODER_MODEL_TYPES):
    #     model_kwargs["attn_implementation"] = "flash_attention_2"

    model = DenseEncoder(args, backward_args, **model_kwargs)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"
    # if model.encoder.config.model_type == "mistral":
    #     model.tokenizer.padding_side = "left"
    print(model.encoder.config)

    args.task_types = [t for t in args.task_types if t.strip()]
    args.tasks = [item for task in args.tasks for item in ([task] if task in TASK_LIST else globals()[task])]
    evaluation = MTEB(
        task_types=args.task_types or None,
        tasks=args.tasks or None,
        task_langs=['en'] if not args.multilingual else None,
    )

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.description['name']
        task_type: str = task_cls.description['type']
        if task_name not in TASK_LIST or task_name in TASK_LIST_RETRIEVAL:
            continue
        if args.dry_run and task_name not in ['Banking77Classification', 'ImdbClassification', 'STS12']:
            continue

        if args.prefix_type == 'query_or_passage':
            prompt: str = 'query: '
        else:
            task_def: str = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
            prompt: str = get_detailed_instruct(task_def)
        model.set_prompt(prompt=prompt)
        logger.info('Set prompt: {}'.format(prompt))

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False
        else:
            model.l2_normalize = True
            logger.info('Set l2_normalize to {}'.format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=['en'] if not args.multilingual else None)
        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))
        eval_splits = ["test"] if "test" in task_cls.description["eval_splits"] else task_cls.description["eval_splits"]
        sub_eval.run(
            model, 
            verbosity=1,
            output_folder=args.output_dir,
            overwrite_results=False,
            eval_splits=eval_splits,
            batch_size=4,
        )


if __name__ == '__main__':
    main()
