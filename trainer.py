from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer)
from pandas import DataFrame, concat
from torch import Tensor,load
from argparse import ArgumentParser, Namespace
from dataset import DGDataModule, DGModel
from distractor import CustomTrainer, CustomT5ForConditionalGeneration
from typing import List, Dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
    parser.add_argument("--input_max_length", type=int, default=512)
    parser.add_argument("--target_max_length", type=int, default=512)
    parser.add_argument("--logs_dir", type=str, default="/content/drive/MyDrive/Thesis/distractor_data/model_1/logs")

    return parser.parse_args()

class Driver():
    def __init__(this,
                 sep_token: str,
                 context_token: str,
                 question_token: str,
                 answer_token:str,
                 batch_size: int,
                 val_batch_size: int, 
                 max_source_token_len: int,
                 max_target_token_len: int):
        this.sep_token = sep_token
        this.context_token = context_token
        this.answer_token = answer_token
        this.question_token = question_token
        this.batch_size = batch_size
        this.val_batch_size = val_batch_size
        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len

    def prepare_distractor_generator_datasets(this,
                                              train_df: DataFrame,
                                              val_df: DataFrame,
                                              test_df: DataFrame,
                                              tokenizer: T5Tokenizer):
        this.dgDataModule = DGDataModule(train_df=train_df,
                                         val_df=val_df,
                                         test_df=test_df,
                                         tokenizer=tokenizer,
                                         sep_token=this.sep_token,
                                         context_token=this.context_token,
                                         question_token=this.question_token,
                                         answer_token=this.answer_token,
                                         batch_size=this.batch_size,
                                         valid_batch_size=this.val_batch_size,
                                         max_source_token_len=this.max_source_token_len,
                                         max_target_token_len=this.max_target_token_len)
        return

    def prepare_distractor_generator_model(this,
                                           model,
                                           new_tokenizer_len: int,
                                           optimizer,
                                           optimizer_lr: float
                                           ):
        this.dgModel = DGModel(model,
                               new_tokenizer_len,
                               optimizer,
                               optimizer_lr)
        return

    def check_is_prepared(this):
        if this.dgModel is None or this.dgDataModule is None:
            raise ValueError("DGModel or DGDataModule not initialized")

    def train_distractor_generator(this,
                                   callbacks: List,
                                   logger,
                                   epochs: int,
                                   accelerator: str = "gpu"):
        this.check_is_prepared()
        this.trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            max_epochs=epochs,
            accelerator=accelerator)
        this.trainer.fit(this.dgModel,
                         this.dgDataModule)

    def val_distractor_generator(this) -> List[Dict]:
        this.check_is_prepared()
        res: List[Dict] = this.trainer.test(this.dgModel,
                                            this.dgDataModule)
        return res

    def test_distractor_generator(this):
        this.check_is_prepared()
        res: List[Dict] = this.trainer.validate(this.dgModel,
                                                this.dgDataModule)
        return res

    def load_dg_model(this,
                      model_path: str,
                      map_location: str = "cpu"):
        if this.dgModel is None:
            raise ValueError("DGModel not initialized")

        this.dgModel.load_state_dict(load(model_path,
                                          map_location=map_location)['state_dict'])
        return

    def run_dg(this,
               train_df: DataFrame,
               val_df: DataFrame,
               test_df: DataFrame,
               tokenizer: T5Tokenizer,

               model,
               new_tokenizer_len: int,
               optimizer,
               optimizer_lr: float,
               callbacks,
               logger,
               epochs: int = 3,
               accelerator: str = "gpu",
               ):
        results: List = []
        this.prepare_distractor_generator_model(model,
                                                new_tokenizer_len,
                                                optimizer,
                                                optimizer_lr,
                                                )
        this.prepare_distractor_generator_datasets(train_df,
                                                   val_df,
                                                   test_df,
                                                   tokenizer)

        results.append(this.train_distractor_generator(callbacks,
                                                       logger,
                                                       epochs,
                                                       accelerator))
        results.append(this.val_distractor_generator())
        results.append(this.test_distractor_generator())

        return results

    def test_dg(this,
                model_path: str,
                model,
                train_df: DataFrame,
                val_df: DataFrame,
                test_df: DataFrame,
                tokenizer: T5Tokenizer,
                new_tokenizer_len: int,
                optimizer,
                optimizer_lr: float,
                map_location: str = "cpu",):
        this.prepare_distractor_generator_model(model, new_tokenizer_len,
                                                optimizer, optimizer_lr)
        this.prepare_distractor_generator_datasets(
            train_df,
            val_df,
            test_df,
            tokenizer)
        this.load_dg_model(model_path,
                           map_location)
        print("Distractor generation model loaded succesfully")

    def generate(this, answer: str,
                 question: str,
                 context: str,
                 incorrect_1: str,
                 incorrect_2: str,
                 incorrect_3: str,
                 tokenizer: T5Tokenizer,
                 num_beams: int = 2,

                 ):
        source_encoding = tokenizer(
            "{} {} {} {} {}".format(answer, this.sep_token,
                                    question, this.sep_token,
                                    context),
            max_length=this.max_source_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        outputs: Tensor = this.dgModel.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            max_length=this.max_target_token_len,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=2.5,
            length_penalty=1.0,
            use_cache=True
        )

        preds = [tokenizer.decode(output,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True) for output in outputs]

        return "".join(preds)

    def show_results(this, generated: str, answer: str,
                     question: str, context: str,
                     incorrect_1: str, incorrect_2: str,
                     incorrect_3: str
                     ):
        print("Context:\n")
        print(context)

        print("Question and answer:")
        print("Question {}\nAnswer: {}".format(question, answer))

        print("========================================Generated Results===========================================================")
        print("Original Distractors: Incorrect-1:{}\nIncorrect-2:{}\nIncorrect-3 {}".format(incorrect_1, incorrect_2, incorrect_3))
        print("Generated: {}".format(generated))
        print("====================================================================================================================")
        print()

    def save_generated_result(this, df: DataFrame,
                              tokenizer: T5Tokenizer,
                              file_name: str,
                              ):
        if this.dgModel is None:
            raise ValueError("DG Model is not initialized")

        result = DataFrame(columns=["correct",
                                    "context",
                                    "question",
                                    "incorrect_1",
                                    "incorrect_2",
                                    "incorrect_3",
                                    "generated"])
        for i in range(len(df)):
            row = df.iloc[i]
            answer = row["correct"]
            context = row["context"]
            question = row["question"]
            incorrect_1 = row["incorrect_1"]
            incorrect_2 = row["incorrect_2"]
            incorrect_3 = row["incorrect_3"]
            generated = this.generate(answer, question,
                                      context, incorrect_1,
                                      incorrect_2, incorrect_3, tokenizer)
            new_row = DataFrame({"correct": answer,
                                 "context": context,
                                 "question": question,
                                 "incorrect_1": incorrect_1,
                                 "incorrect_2": incorrect_2,
                                 "incorrect_3": incorrect_3,
                                 "generated": generated},
                                index=[0])

            result = concat([result, new_row], ignore_index=True)

        result.to_csv(file_name, index=False)
        print("Results saved to {}".format(file_name))

    def try_generate(this, tokenizer,
                     df: DataFrame, n: int):
        if this.dgModel is None:
            raise ValueError("DGModel is not initialized")

        for i in range(n):
            row = df.iloc[i]
            answer = row["correct"]
            context = row["context"]
            question = row["question"]
            incorrect_1 = row["incorrect_1"]
            incorrect_2 = row["incorrect_2"]
            incorrect_3 = row["incorrect_3"]

            generated = this.generate(answer, question, context,
                                      incorrect_1, incorrect_2,
                                      incorrect_3, tokenizer)

            this.show_results(generated, answer, question,
                              context, incorrect_1, incorrect_2,
                              incorrect_3)
if __name__ == "__main__":
    args: Namespace = parse_argument()
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [args.question_token, args.answer_token,
                                      args.context_token, args.distractor_token]
    })
    driv = Driver(args.distractor_token, args.context_token,
                  args.question_token, args.answer_token,
                  args.train_batch_size, args.valid_batch_size,
                  args.input_max_length, args.target_max_length
                  )
    model = CustomT5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    data = datasets.load_dataset(args.dataset_path)
    train_data = data["train"]
    valid_data = data["validation"]
    test_data = data["test"]
    #driv.prepare_distractor_generator_model(model, len(tokenizer), AdamW(), args.lr)
    #driv.prepare_distractor_generator_datasets(DataFrame(train_data), DataFrame(valid_data), DataFrame(test_data))
    model_callbacks = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="best-checkpoint",
        monitor="val_loss",
        save_last=False,
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger(args.logs_dir, name="distractor_log")
    driv.run_dg(
        DataFrame(train_data),
        DataFrame(valid_data),
        DataFrame(test_data),
        tokenizer,
        model,
        len(tokenizer),
        AdamW(model.parameters()),
        args.lr,
        callbacks=model_callbacks,
        logger=logger
    )
 

