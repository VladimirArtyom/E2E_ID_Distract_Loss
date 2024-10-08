from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import List
import torch

import torch.nn.functional as fu
from torch import stack, exp, log

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, labels=None, correct_answer_ids=None, **kwargs):
        # Call the parent forward method
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        return outputs  # Return the usual output for Trainer to handle
class CustomTrainer(Trainer):

    def get_train_dataloader(this) -> DataLoader:
        return DataLoader(this.train_dataset, batch_size=this.args.per_device_train_batch_size ,pin_memory=False)
 
    def get_eval_dataloader(this, eval_dataset=None) -> DataLoader:
        if eval_dataset is None:
            eval_dataset = this.eval_dataset
        return DataLoader(eval_dataset, batch_size=this.args.per_device_eval_batch_size, pin_memory=False)
    """
    def compute_loss(this, model, inputs, return_outputs=False):
        # Forward pass to get the model outputs
        outputs = model(**inputs)
        logits = outputs.logits  # Get logits from model output
        
        # Tokenize and extract the correct answer from inputs
        sep_token_id = 32103
        epsilon = 1e-8
        correct_answer_logits = logits[:, 0, :]  # Logits for the correct answer
        #correct_answer_logits = inputs["correct_answer_ids"].float()

        # Extract distractor logits
        distractor_logits = []  
        for batch_indx in range(logits.size(0)):
            batch_distractors = []
            sep_positions = (inputs['labels'][batch_indx] == sep_token_id).nonzero(as_tuple=True)
            for pos in sep_positions[0]:
                if pos + 1 < logits.size(1):  # Ensure there's a token after <sep>
                    batch_distractors.append(logits[batch_indx, pos + 1, :])  # Collect distractor logits
            distractor_logits.append(torch.stack(batch_distractors))
        # Stack the distractor logits into a tensor
        distractor_logits = torch.stack(distractor_logits)#.view(logits.size(0), -1, logits.size(2))  # Shape: (batch_size, num_distractors, vocab_size)
        
        # Normalize the logits for cosine similarity
        correct_answer_norm = fu.normalize(correct_answer_logits, p=2, dim=1)  # Normalize along the token dimension
        distractor_norm = fu.normalize(distractor_logits, p=2, dim=2)
        # Compute cosine similarity between correct answer and distractors
        sim_correct_to_distractors = fu.cosine_similarity(correct_answer_norm.unsqueeze(1), distractor_norm, dim=2)  # Shape: (batch_size, num_distractors)

        # We want distractors to be different from the correct answer (minimize similarity)
        # Therefore, loss should maximize the distance between the correct answer and the distractors
        distractor_loss = torch.relu(-torch.log(torch.clamp(1 - sim_correct_to_distractors.mean(dim=1) , min=epsilon)))  # Minimize similarity with correct answer
        # Compute similarity between distractors themselves (to ensure plausibility)
        sim_distractors = fu.cosine_similarity(distractor_norm[:, 0, :].unsqueeze(1), distractor_norm[:, 1:, :], dim=2)  # Shape: (batch_size, num_distractors-1)
        
        # Loss to encourage distractors to be similar to each other (plausibility)
        plausibility_loss = torch.relu(-torch.log(torch.clamp(sim_distractors.mean(dim=1) + epsilon, min=epsilon)))  # Maximize similarity between distractors

        # Total loss is the sum of both terms
        loss = distractor_loss.mean() + plausibility_loss.mean()

        return (loss, outputs) if return_outputs else loss
    """