import argparse

import torch

from codebook_features import evaluation, models, run_clm

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument(
    "--model_name_or_path", type=str, default="EleutherAI/pythia-70m-deduped"
)
parser.add_argument("--pretrained_path", type=str, default="taufeeque/best-cb-model")
parser.add_argument("--hooked_transformer", type=bool, default=False)
parser.add_argument("--eval_on", type=str, default="train")
parser.add_argument("--max_samples", type=int, default=10_000)
parser.add_argument("--dataset_name", type=str, default="wikitext")
parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-v1")
parser.add_argument("--cache_dir", type=str, default="/data/.cache/huggingface/")
parser.add_argument("--save_dir", type=str, default="/homedir")
parser.add_argument("--codebooks", type=bool, default=True)
parser.add_argument("--logging", type=bool, default=False)

args = parser.parse_args()

model = models.wrap_codebook(
    model_or_path=args.model_name_or_path, pretrained_path=args.pretrained_path
)
if args.hooked_transformer:
    model = models.convert_to_hooked_model(args.model_name_or_path, model)
if not args.logging:
    model.disable_logging()
model.eval().to("cuda")
model.set_hook_kwargs(cosine=True)

if args.codebooks:
    model.enable_codebooks()
else:
    model.disable_codebooks()
# model = torch.compile(model)

model_args = run_clm.ModelArguments(
    model_name_or_path=args.model_name_or_path, cache_dir=args.cache_dir
)
data_args = run_clm.DataTrainingArguments(
    dataset_name=args.dataset_name,
    dataset_config_name=args.dataset_config_name,
    streaming=False,
    max_train_samples=args.max_samples,
    max_eval_samples=args.max_samples,
)
tokens, cb_acts, metrics, output_dir = evaluation.evaluate(
    model=model,
    model_args=model_args,
    data_args=data_args,
    eval_on=args.eval_on,
    save_dir=args.save_dir,
)

print("Output dir:", output_dir)

with open(f"{output_dir}/info.txt", "w") as f:
    f.write(f"Model: {args.model_name_or_path}\n")
    f.write(f"Pretrained path: {args.pretrained_path}\n")
    f.write(f"Hooked transformer: {args.hooked_transformer}\n")
    f.write(f"Eval on: {args.eval_on}\n")
    f.write(f"Max samples: {args.max_samples}\n")
    f.write(f"Dataset name: {args.dataset_name}\n")
    f.write(f"Dataset config name: {args.dataset_config_name}\n")
    f.write(f"Save dir: {args.save_dir}\n")
