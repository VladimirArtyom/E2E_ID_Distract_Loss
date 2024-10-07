from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer)
from torch import Tensor
from argparse import ArgumentParser, Namespace
from dataset import DistractorDataset
import datasets

def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="Voslannack/race_id")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    ### Model identifier
    parser.add_argument("--distractor_token", type=str, default="<sep>")
    parser.add_argument("--question_token", type=str, default="<question>")
    parser.add_argument("--answer_token", type=str, default="<answer>")
    parser.add_argument("--context_token", type=str, default="<context>")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)


if __name__ == "__main__":
    args: Namespace = parse_argument()
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [args.question_token, args.answer_token,
                                      args.context_token, args.distractor_token]
    })
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    data = datasets.load_dataset(args.dataset_path)
 
    train_data = data["train"]
    valid_data = data["valid"]
    test_data = data["test"]

    train_dataset = DistractorDataset(data=train_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_mask=args.context_token,
                                      )
    valid_dataset = DistractorDataset(data=valid_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_mask=args.context_token,
                                      )
    test_dataset = DistractorDataset(data=test_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_mask=args.context_token,
                                      )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_gpu_train_batch_size=args.train_batch_size,
        per_gpu_eval_batch_size=args.valid_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset
    )

