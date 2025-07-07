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

        self.l2_normalize = False
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
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding'):
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

    @torch.no_grad()
    def encode_words(self, dataset, **kwargs) -> np.ndarray:
        encoded_embeds = []
        labels = []
        for idx in tqdm.tqdm(range(0, len(dataset)), desc='encoding'):
            batch_inputs = dataset[idx]

            batch_dict = {
                'input_ids': torch.tensor([batch_inputs['input_ids']], dtype=torch.long),
                'attention_mask': torch.tensor([batch_inputs['attention_mask']], dtype=torch.long),
            }
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs = self.encoder(**batch_dict)
                embeds = outputs[self.keyword][0]

                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().to(torch.float32).numpy())
                labels.extend(batch_inputs['labels'])

        encoded_embeds = np.concatenate(encoded_embeds, axis=0)
        labels = np.array(labels)
        valid_indices = labels != -100

        encoded_embeds = encoded_embeds[valid_indices]
        labels = labels[valid_indices]
        return encoded_embeds, labels

    def set_prompt(self, prompt: str):
        self.prompt = prompt

@dataclass
class TaskArgument:
    model_name_or_path: str = field(default="outputs/base/tinyllama-1.1B", metadata={"help": "which model to use"})
    tasks: List[str] = field(default_factory=list, metadata={"help": "domain-specific tasks to evaluate"})
    output_dir: str = field(default="outputs/domain", metadata={"help": "output directory"})
    pool_type: str = field(default="avg", metadata={"help": "pool type"})
    prefix_type: str = field(default="query_or_passage", metadata={"help": "prefix type"})

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
    classifier = Classifier()

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"
    # if model.encoder.config.model_type == "mistral":
    #     model.tokenizer.padding_side = "left"
    print(model.encoder.config)

    args.tasks = [item for task in args.tasks for item in ([task] if task in DOMAIN_TASK_LIST else globals()[task])]

    for task_name in args.tasks:
        logger.info(f"Running {task_name} Task")
        if args.prefix_type == 'query_or_passage':
            prompt: str = 'query: '
        else:
            task_def: str = get_task_def_by_task_name_and_type(task_name=task_name, task_type="Domain")
            prompt: str = get_detailed_instruct(task_def)
            model.set_prompt(prompt=prompt)
            logger.info('Set prompt: {}'.format(prompt))

        dataset = get_task_dataset(task_name, model.tokenizer)
        results = None

        if task_name == "MQP":
            logger.info('Set l2_normalize to True for STS task')
            model.l2_normalize = True

            encoded_embeds_1  = model.encode(dataset["sentence_1"])
            encoded_embeds_2  = model.encode(dataset["sentence_2"])
            preds = np.sum(encoded_embeds_1 * encoded_embeds_2, axis=-1)
            _, accuracy = find_best_accuracy(preds, dataset["label"])
            results = {"accuracy": accuracy}
        elif task_name == "NER":
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False

            encoded_embeds, labels  = model.encode_words(dataset)
            results = classifier(encoded_embeds, labels, diable_smote=True)
        else:
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False

            encoded_embeds  = model.encode(dataset["sentence"])
            results = classifier(encoded_embeds, dataset["label"])

        print(f"{task_name} Results:", results)
        with open(f"{args.output_dir}/{task_name}.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
