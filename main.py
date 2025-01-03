import time
import torch
import itertools
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig
from fvcore.nn import FlopCountAnalysis

import wandb

######################################################
# MODEL & TOKENIZER LOADING
######################################################
model_name = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(r=8)
model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

######################################################
# PREPROCESS / FLATTEN / TOKENIZE LOGIC
######################################################
def preprocess_conversation(example):
    """
    Flattens a single conversation (example["messages"]) into
    multiple user->assistant turns, returning lists of inputs/outputs.
    """
    dialogue = ""
    inputs = []
    outputs = []
    for message in example["messages"]:
        if message["role"] == "user":
            dialogue += f"User: {message['content']} "
        elif message["role"] == "assistant":
            # Each assistant message pairs with what came before
            inputs.append(dialogue.strip())
            outputs.append(f"Assistant: {message['content']}")
            dialogue += f"Assistant: {message['content']} "
    return inputs, outputs

def tokenize_pair(user_text, assistant_text):
    """
    Tokenizes a single (input, output) pair with padding/truncation,
    then replaces pad_token_id in labels with -100.
    """
    tokenized_inputs = tokenizer(
        user_text, padding="max_length", truncation=True, max_length=512
    )
    tokenized_outputs = tokenizer(
        assistant_text, padding="max_length", truncation=True, max_length=512
    )
    labels = tokenized_outputs["input_ids"]
    labels = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in labels
    ]

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

######################################################
# ITERABLE DATASET
######################################################
class UltraChatIterableDataset(IterableDataset):
    def __init__(self, hf_stream):
        """
        hf_stream: a streaming dataset from load_dataset(..., streaming=True)
        """
        super().__init__()
        self.hf_stream = hf_stream

    def __iter__(self):
        for raw_example in self.hf_stream:
            # Flatten the conversation
            inps, outs = preprocess_conversation(raw_example)
            # For each user->assistant turn, tokenize and yield
            for user_text, assistant_text in zip(inps, outs):
                tokenized = tokenize_pair(user_text, assistant_text)
                yield {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["labels"]
                }

######################################################
# LIMITING DATASETS & DATALOADERS
######################################################
class LimitDataset(IterableDataset):
    def __init__(self, base_dataset, limit):
        super().__init__()
        self.base_dataset = base_dataset
        self.limit = limit

    def __iter__(self):
        count = 0
        for item in self.base_dataset:
            if count >= self.limit:
                break
            yield item
            count += 1

train_stream_all = load_dataset("HuggingFaceH4/ultrachat_200k", split='train_sft', streaming=True)
test_stream = load_dataset("HuggingFaceH4/ultrachat_200k", split='test_sft', streaming=True)
train_stream_all = train_stream_all.shuffle(seed=42, buffer_size=10_000)
test_stream = test_stream.shuffle(seed=42, buffer_size=10_000)
validation_stream = train_stream_all.take(30)
train_stream = train_stream_all.skip(30)

train_set = UltraChatIterableDataset(train_stream)
test_set = UltraChatIterableDataset(test_stream)
validation_set = UltraChatIterableDataset(validation_stream)

# train_set = LimitDataset(train_set_full, limit=50)
# validation_set = LimitDataset(validation_set_full, limit=10)
# test_set = LimitDataset(test_set_full, limit=20)

def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(train_set, batch_size=30, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_set, batch_size=30, shuffle=False, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_set, batch_size=30, shuffle=False, collate_fn=collate_fn)

######################################################
# FLOP-RELATED HELPER FUNCTIONS
######################################################
def compute_flops(model, batch, mode='evaluation'):
    try:
        analysis = FlopCountAnalysis(model, batch["input_ids"])
        analysis.tracer_warnings('none')  # Suppress warnings
        forward_flops = analysis.total()
        if mode == 'training':
            # Approx. 2x for backward
            backward_flops = 2 * forward_flops
            return forward_flops + backward_flops
        elif mode == 'evaluation':
            return forward_flops
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    except Exception as e:
        print(f"FLOP computation failed: {e}")
        return 0

def train_step(model, batch, optimizer):
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

def compute_avg_test_loss(model, test_dataloader, device):
    total_test_loss = 0
    total_batches = 0
    model.eval()
    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = {k: v.to(device) for k, v in test_batch.items()}
            outputs = model(
                input_ids=test_batch['input_ids'],
                attention_mask=test_batch['attention_mask'],
                labels=test_batch['labels']
            )
            total_test_loss += outputs.loss
            total_batches += 1
    model.train()
    if total_batches == 0:
        return torch.tensor(0.0, device=device)
    return total_test_loss / total_batches

######################################################
# VANILLA TRAINING WITH W&B LOGGING
######################################################
def vanilla_train(model, train_dataloader, test_dataloader, num_epochs=5, device='cuda'):
    """
    Runs a basic (vanilla) training loop and logs key metrics to Weights & Biases:
        - Train Loss
        - Test Loss
        - FLOPs (cumulative)
        - Time
    """

    # 1) Initialize Weights & Biases
    wandb.init(project="my_cluster_training", name="vanilla_train_run")
    wandb.config.update({
        "learning_rate": 1e-4,
        "optimizer": "SGD",
        "momentum": 0.9,
        "num_epochs": num_epochs,
        "batch_size": train_dataloader.batch_size
    })

    test_losses_overall = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    model.train()

    # Safely try to fetch one batch for FLOP computation
    train_iter = iter(train_dataloader)
    sample_train_batch = next(train_iter, None)
    if sample_train_batch is not None:
        sample_train_batch = {k: v.to(device) for k, v in sample_train_batch.items()}
        train_flops = compute_flops(model, sample_train_batch, mode="training")
    else:
        train_flops = 0

    total_flops = 0
    start_time = time.time()

    global_step = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        batch_idx = 0

        # Re-initialize iterator for this epoch (streaming)
        train_iter = iter(train_dataloader)
        while True:
            batch = next(train_iter, None)
            if batch is None:
                # End of streaming dataset
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            # Accumulate FLOPs
            total_flops += train_flops

            loss = train_step(model, batch, optimizer)
            total_train_loss += loss.item()
            batch_idx += 1
            global_step += 1

            # Compute test loss after each step
            avg_test_loss = compute_avg_test_loss(model, test_dataloader, device).item()
            test_losses_overall.append(avg_test_loss)

            # 2) Log metrics to wandb
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": avg_test_loss,
                "epoch": epoch + 1,
                "step": global_step,
                "total_flops": total_flops,  # raw number of FLOPs
                "tf_flops": total_flops / 1e12,
            })

        if batch_idx > 0:
            avg_train_loss = total_train_loss / batch_idx
        else:
            avg_train_loss = 0.0

    total_time = time.time() - start_time

    # 3) Log final metrics to wandb
    wandb.log({
        "final_test_loss": avg_test_loss,
        "training_time_s": total_time,
        "final_total_flops": total_flops,
        "final_tf_flops": total_flops / 1e12,
    })

    # 4) Close out this wandb run
    wandb.finish()

    return avg_test_loss, test_losses_overall, total_time, total_flops

######################################################
# FAST FORWARD TRAINING WITH W&B LOGGING
######################################################
def fast_forward_step(model, delta_weights):
    for name, param in model.named_parameters():
        param.data.add_(delta_weights[name])
    return model

def ff_train(model,
             train_dataloader,
             test_dataloader,
             validation_batch,
             final_vanilla_loss,
             Tinterval=5,
             device='cuda'):
    """
    Implements "fast forward" training and logs key metrics to Weights & Biases.
    """
    # 1) Initialize a separate W&B run (or reuse same project with a different name)
    wandb.init(project="my_cluster_training", name="fast_forward_run")
    wandb.config.update({
        "learning_rate": 1e-4,
        "optimizer": "SGD",
        "momentum": 0.9,
        "Tinterval": Tinterval
    })

    test_losses_overall = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    model.train()
    avg_test_loss = float('inf')
    prev_val_loss = float('inf')
    step_count = 0

    validation_batch = {k: v.to(device) for k, v in validation_batch.items()}

    # Precompute FLOPs for training and validation
    train_iter = iter(train_dataloader)
    sample_train_batch = next(train_iter, None)
    if sample_train_batch is not None:
        sample_train_batch = {k: v.to(device) for k, v in sample_train_batch.items()}
        train_flops = compute_flops(model, sample_train_batch, mode="training")
    else:
        train_flops = 0

    val_flops = compute_flops(model, validation_batch, mode="evaluation")

    total_flops = 0
    start_time = time.time()

    # We'll re-create an iterator for Tinterval steps
    while avg_test_loss > final_vanilla_loss - 0.0001:
        # 1) SGD phase for Tinterval steps
        prev_weights = {name: p.clone() for name, p in model.named_parameters()}

        train_iter = iter(train_dataloader)
        for i in range(Tinterval):
            batch = next(train_iter, None)
            if batch is None:
                # No more data in streaming
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            total_flops += train_flops
            loss = train_step(model, batch, optimizer)

            avg_test_loss = compute_avg_test_loss(model, test_dataloader, device).item()
            test_losses_overall.append(avg_test_loss)
            step_count += 1

            # 2) Log after each training step
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": avg_test_loss,
                "step_count": step_count,
                "total_flops": total_flops,
                "tf_flops": total_flops / 1e12
            })

        # 2) Compute delta_w
        delta_weights = {}
        for name, param in model.named_parameters():
            delta_weights[name] = param.clone() - prev_weights[name]

        # 3) Fast Forward stage
        while True:
            fast_forward_step(model, delta_weights)
            total_flops += val_flops

            avg_test_loss = compute_avg_test_loss(model, test_dataloader, device).item()
            test_losses_overall.append(avg_test_loss)

            wandb.log({
                "test_loss_ff": avg_test_loss,
                "step_count": step_count,
                "total_flops": total_flops,
                "tf_flops": total_flops / 1e12
            })

            if avg_test_loss <= final_vanilla_loss - 0.0001:
                break

            # Compute validation loss
            with torch.no_grad():
                outputs = model(
                    input_ids=validation_batch['input_ids'],
                    attention_mask=validation_batch['attention_mask'],
                    labels=validation_batch['labels']
                )
                val_loss = outputs.loss.item()

            if val_loss >= prev_val_loss:
                break

            prev_val_loss = val_loss

        if avg_test_loss <= final_vanilla_loss - 0.0001:
            break

    total_time = time.time() - start_time

    # Final logs
    wandb.log({
        "final_ff_loss": avg_test_loss,
        "training_time_s": total_time,
        "final_total_flops": total_flops,
        "final_tf_flops": total_flops / 1e12,
    })
    wandb.finish()

    return avg_test_loss, test_losses_overall, total_time, total_flops


######################################################
# TRAINING
######################################################
if __name__ == "__main__":
    vanilla_final_loss, vanilla_test_curve, vanilla_time, vanilla_flops = vanilla_train(
        model, train_dataloader, test_dataloader, num_epochs=2, device=device
    )

    # Prepare one validation batch for ff_train
    validation_iter = iter(validation_dataloader)
    validation_batch = next(validation_iter, None)
    if validation_batch is None:
        raise ValueError("No validation data found!")

    ff_final_loss, ff_test_curve, ff_time, ff_flops = ff_train(
        model,
        train_dataloader,
        test_dataloader,
        validation_batch,
        final_vanilla_loss=vanilla_final_loss,
        Tinterval=6,  # do a few steps before each "fast forward" attempt
        device=device
    )
