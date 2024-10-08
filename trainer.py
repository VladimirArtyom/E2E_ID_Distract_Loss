from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer)
from torch import Tensor
from argparse import ArgumentParser, Namespace
from dataset import DistractorDataset
from distractor import CustomLossTrainer, CustomT5ForConditionalGeneration
import datasets

def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="Voslannack/race_id")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    ### Model identifier
    parser.add_argument("--distractor_token", type=str, default="<sep>")
    parser.add_argument("--question_token", type=str, default="<question>")
    parser.add_argument("--answer_token", type=str, default="<answer>")
    parser.add_argument("--context_token", type=str, default="<context>")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Thesis/distractor_data/model_1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logs_dir", type=str, default="/content/drive/MyDrive/Thesis/distractor_data/model_1/logs")

    return parser.parse_args()

if __name__ == "__main__":
    args: Namespace = parse_argument()
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [args.question_token, args.answer_token,
                                      args.context_token, args.distractor_token]
    })
    model = CustomT5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    data = datasets.load_dataset(args.dataset_path)
 
    train_data = data["train"]
    valid_data = data["validation"]
    test_data = data["test"]

    train_dataset = DistractorDataset(data=train_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_token=args.context_token,
                                      question_token=args.question_token,
                                      sep_token=args.distractor_token,
                                      answer_token=args.answer_token,
                                      device=args.device
                                      )
    valid_dataset = DistractorDataset(data=valid_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_token=args.context_token,
                                      question_token=args.question_token,
                                      sep_token=args.distractor_token,
                                      answer_token=args.answer_token,
                                      device=args.device
                                      )

    test_dataset = DistractorDataset(data=test_data, max_length=args.max_length,
                                      tokenizer=tokenizer, context_token=args.context_token,
                                      question_token=args.question_token,
                                      sep_token=args.distractor_token,
                                      answer_token=args.answer_token,
                                      device=args.device
                                      )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs"
    )

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    #print(tokenizer.convert_tokens_to_ids("<sep>"))
    trainer.train()
