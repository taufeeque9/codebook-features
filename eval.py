from codebook_features import models, run_clm, evaluation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-70m-deduped")
parser.add_argument("--pretrained_path", type=str, default="taufeeque/best-cb-model")
parser.add_argument("--hooked_transformer", type=bool, default=False)
parser.add_argument("--eval_on", type=str, default="train")
parser.add_argument("--max_train_samples", type=int, default=10_000)

args = parser.parse_args()

model = models.wrap_codebook(model_or_path=args.model_name_or_path, pretrained_path=args.pretrained_path)
if args.hooked_transformer:
    model = models.convert_to_hooked_model(args.model_name_or_path, model)

model_args = run_clm.ModelArguments(model_name_or_path=args.model_name_or_path)
data_args = run_clm.DataTrainingArguments(
    dataset_name="wikitext",
    dataset_config_name="wikitext-103-v1",
    streaming=False,
    max_train_samples=args.max_train_samples,
)
tokens, cb_acts, metrics = evaluation.evaluate(
    model=model, model_args=model_args, data_args=data_args, eval_on=args.eval_on,
)
