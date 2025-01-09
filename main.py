import os
import time
import torch
import argparse
import itertools
import copy

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig
from fvcore.nn import FlopCountAnalysis

import wandb

import logging
import warnings

# Hide Python warnings
warnings.filterwarnings("ignore")

# Hide fvcore logs below CRITICAL
logging.getLogger("fvcore.nn").setLevel(logging.CRITICAL)

######################################################
# MODEL & TOKENIZER LOADING (GLOBAL)
######################################################
model_name = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(r=8)
base_model = get_peft_model(base_model, lora_config)

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
            inputs.append(dialogue.strip())
            outputs.append(f"Assistant: {message['content']}")
            dialogue += f"Assistant: {message['content']} "
    return inputs, outputs

def tokenize_pair(user_text, assistant_text, max_len=512):
    """
    Tokenizes a single (input, output) pair with padding/truncation,
    then replaces pad_token_id in labels with -100.
    """
    tokenized_inputs = tokenizer(
        user_text, padding="max_length", truncation=True, max_length=max_len
    )
    tokenized_outputs = tokenizer(
        assistant_text, padding="max_length", truncation=True, max_length=max_len
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
            inps, outs = preprocess_conversation(raw_example)
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

def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model)

######################################################
# FLOP-RELATED HELPER FUNCTIONS
######################################################
def compute_flops(model, batch, device, mode='evaluation'):
    try:
        input_ids, _, _ = batch
        analysis = FlopCountAnalysis(model, input_ids.to(device))
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

def train_step(model, batch, optimizer, device):
    input_ids, attention_mask, labels = batch
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device)
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def compute_avg_loss(model, dataloader, device):
    total_batches = 0
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device)
            )
            total_loss += outputs.loss.item()
            total_batches += 1
    model.train()
    if total_batches == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / total_batches

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
    # Only log from rank=0 (main process) to avoid duplicates
    if dist.is_initialized():
        is_main_process = (dist.get_rank() == 0)
    else:
        is_main_process = True

    if is_main_process:
        wandb.init(project="ff-lora-pythia1.4b-ultrachat", name="vanilla_train_run")
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

    # Compute FLOPs for a sample batch
    sample_train_batch = next(iter(train_dataloader), None)
    if sample_train_batch is not None:
        train_flops = compute_flops(model, sample_train_batch, device, mode="training")
    else:
        train_flops = 0

    total_tflops = 0
    start_time = time.time()
    global_step = -1
    avg_test_loss = 0.0  # just to have a default

    for epoch in range(num_epochs):
        batch_idx = 0

        for batch in train_dataloader:
            total_tflops += train_flops / 1e12
            loss = train_step(model, batch, optimizer, device)

            batch_idx += 1
            global_step += 1

            # Test loss each step
            if global_step % 100 == 0:
                avg_test_loss = compute_avg_loss(model, test_dataloader, device)
            test_losses_overall.append(avg_test_loss)

            # Log metrics
            if is_main_process:
                wandb.log({
                    "train_loss": loss,
                    "test_loss": avg_test_loss,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_flops": total_tflops,
                })

    total_time = time.time() - start_time
    if is_main_process:
        wandb.log({
            "final_test_loss": avg_test_loss,
            "training_time_s": total_time,
            "final_total_flops": total_tflops,
        })
        wandb.finish()

    return avg_test_loss, test_losses_overall, total_time, total_tflops

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
             validation_dataloader,
             final_vanilla_loss,
             Tinterval=5,
             device='cuda'):
    """
    Implements "fast forward" training and logs key metrics to Weights & Biases.
    """
    if dist.is_initialized():
        is_main_process = (dist.get_rank() == 0)
    else:
        is_main_process = True

    if is_main_process:
        wandb.init(project="ff-lora-pythia1.4b-ultrachat", name="fast_forward_run")
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

    sample_validation_batch = next(iter(validation_dataloader), None)
    val_flops = compute_flops(model, sample_validation_batch, device, mode="evaluation")

    # Precompute FLOPs for training and validation
    sample_train_batch = next(iter(train_dataloader), None)
    if sample_train_batch is not None:
        sample_train_batch = {k: v.to(device) for k, v in sample_train_batch.items()}
        train_flops = compute_flops(model, sample_train_batch, device, mode="training")
    else:
        train_flops = 0

    total_tflops = 0
    start_time = time.time()

    while avg_test_loss > final_vanilla_loss - 0.0001:
        # SGD phase for Tinterval steps
        prev_weights = {name: p.clone() for name, p in model.named_parameters()}

        num_steps = 0
        for batch in train_dataloader:
            if num_steps == Tinterval:
                break

            total_tflops += train_flops / 1e12
            loss = train_step(model, batch, optimizer, device)

            avg_test_loss = compute_avg_loss(model, test_dataloader, device)
            test_losses_overall.append(avg_test_loss)
            step_count += 1

            if is_main_process:
                wandb.log({
                    "train_loss": loss,
                    "test_loss": avg_test_loss,
                    "step_count": step_count,
                    "total_flops": total_tflops,
                })
            num_steps += 1

        # Compute delta_w
        delta_weights = {}
        for name, param in model.named_parameters():
            delta_weights[name] = param.clone() - prev_weights[name]

        # Fast Forward stage
        while True:
            fast_forward_step(model, delta_weights)
            total_tflops += val_flops / 1e12

            avg_test_loss = compute_avg_loss(model, test_dataloader, device)
            test_losses_overall.append(avg_test_loss)

            if is_main_process:
                wandb.log({
                    "test_loss_ff": avg_test_loss,
                    "step_count": step_count,
                    "total_flops": total_tflops,
                })

            if avg_test_loss <= final_vanilla_loss - 0.0001:
                break

            with torch.no_grad():
                val_loss = compute_avg_loss(model, validation_dataloader, device)

            if val_loss >= prev_val_loss:
                break
            prev_val_loss = val_loss

        if avg_test_loss <= final_vanilla_loss - 0.0001:
            break

    total_time = time.time() - start_time
    if is_main_process:
        wandb.log({
            "final_ff_loss": avg_test_loss,
            "training_time_s": total_time,
            "final_total_flops": total_tflops,
        })
        wandb.finish()

    return avg_test_loss, test_losses_overall, total_time, total_tflops


######################################################
# TRAIN FUNCTION (EACH PROCESS)
######################################################
def train_process(local_rank, args):
    """
    Each process (rank) will run this function. We:
      1) Initialize the process group
      2) Set up local device
      3) (Single-process) load & shuffle data
      4) Wrap the model in DDP
      5) Run training
    """
    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Copy the model to this device
    model = base_model.to(device)

    #  Load & shuffle dataset
    train_stream_all = load_dataset("HuggingFaceH4/ultrachat_200k",
                                    split='train_sft',
                                    streaming=True)

    test_stream_all = load_dataset("HuggingFaceH4/ultrachat_200k",
                                   split='test_sft',
                                   streaming=True)

    train_stream_all = train_stream_all.shuffle(seed=42, buffer_size=1000)

    test_stream_all = test_stream_all.shuffle(seed=42, buffer_size=1000)

    # For validation, let's take the first 32 from the train
    validation_stream = train_stream_all.take(32)
    # Then skip those so we don't re-train on them
    train_stream = train_stream_all.skip(32)
    # Test set is 1000 long
    test_stream = test_stream_all.take(1000)

    # Wrap into IterableDatasets
    train_set = UltraChatIterableDataset(train_stream)
    test_set = UltraChatIterableDataset(test_stream)
    validation_set = UltraChatIterableDataset(validation_stream)

    # Limit data for demonstration
    # train_set = LimitDataset(train_set, limit=20000)

    # Build DataLoaders
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=1, pin_memory=True)

    # Wrap vanilla model in DDP
    model_vanilla = copy.deepcopy(model)
    model_vanilla = DDP(model_vanilla, device_ids=[local_rank], output_device=local_rank)

    # Run vanilla train
    vanilla_final_loss, _, _, _ = vanilla_train(
        model_vanilla, train_dataloader, test_dataloader, num_epochs=args.num_epochs, device=device
    )

    # Wrap FF model in DDP
    model_ff = copy.deepcopy(model)
    model_ff = DDP(model_ff, device_ids=[local_rank], output_device=local_rank)

    # Run FF train
    if vanilla_final_loss > 0:
        ff_train(
                model_ff,
                train_dataloader,
                test_dataloader,
                validation_dataloader,
                final_vanilla_loss=vanilla_final_loss,
                Tinterval=6,
                device=device
            )
    else:
        print(f"Error during vanilla training. Final loss of {vanilla_final_loss} encountered!")

    # Clean up
    dist.destroy_process_group()

######################################################
# MAIN ENTRYPOINT
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {local_rank}] Starting training with {args.num_gpus} GPUs total...")

    train_process(local_rank, args)

if __name__ == "__main__":
    main()
