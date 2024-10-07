from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch import Tensor
from typing import List
import torch

import torch.nn.functional as fu
from torch import stack, exp, log
class CustomLossTrainer(Trainer):
    def __init__(this, tokenizer, temperature, *args, **kwargs):
        super().__init__(args, kwargs=kwargs)
        this.tokenizer = tokenizer
        this.temperature = temperature
    def compute_loss(this, model, inputs, return_outputs=False):
        # Forward pass to get the model outputs
        outputs = model(**inputs)
        logits = outputs.logits  # Get logits from model output
        
        # Tokenize and extract the correct answer from inputs
        sep_token_id = this.tokenizer.convert_tokens_to_ids('<sep>')
        correct_answer_logits = logits[:, 0, :]  # Logits for the correct answer

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
        distractor_loss = -torch.log(1 - sim_correct_to_distractors.mean(dim=1))  # Minimize similarity with correct answer
        
        # Compute similarity between distractors themselves (to ensure plausibility)
        sim_distractors = fu.cosine_similarity(distractor_norm[:, 0, :], distractor_norm[:, 1:, :], dim=2)  # Shape: (batch_size, num_distractors-1)
        
        # Loss to encourage distractors to be similar to each other (plausibility)
        plausibility_loss = -torch.log(sim_distractors.mean(dim=1))  # Maximize similarity between distractors

        # Total loss is the sum of both terms
        loss = distractor_loss.mean() + plausibility_loss.mean()

        return (loss, outputs) if return_outputs else loss