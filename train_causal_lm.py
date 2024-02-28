import argparse
import os
import math
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

'''
Training LLM using Huggingface Accelerate + Deepspeed.
'''

def parse_option():
    parser = argparse.ArgumentParser()
    
    # global args
    parser.add_argument('--per_device_train_batch_size', type = int, default = 4,
                        help = 'batch size per gpu device.')
    parser.add_argument('--block_size', type = int, default = 8192,
                        help = 'block size, i.e., the length of training sequences.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pretrained_model_name_or_path', type = str, default = "bigcode/starcoder")
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 5e-5, help = "5e-5 for pre-training, 5e-6 for fine-tuning.")
    parser.add_argument('--warmup_ratio', type = float, default = 0.0, help = "ratio of total training steps used for a linear warmup from 0 to max lr.")
    parser.add_argument('--checkpointing_steps', type = int, default = 300)
    parser.add_argument('--tensorboard_log_dir', type = str, default = "./train_logs")
    parser.add_argument('--mode', type = str, default = "pt")
    parser.add_argument('--output_ckpt_dir', type = str, default = "./ckpts")
    parser.add_argument('--save_all_states', action = 'store_true', 
        help = "whether to save states of model, optimizer, and lr scheduler for resuming training, otherwise only model states are saved.")

    # args for pre-training
    parser.add_argument('--pt_data_dir', type = str, default = "./data/corpus.bin")
    parser.add_argument('--resume_from_checkpoint', type = str, default = None, 
                            help = "resuming pre-training from a checkpoint")
    parser.add_argument('--resume_tag', type = str, default = None)
    
    # args for supervised fine-tuning
    parser.add_argument('--text2sql_data_dir', type = str, default = "./data/sft_train_text2sql.json")
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)
    
    opt = parser.parse_args()

    return opt

def checkpoint_model_optimizer_scheduler(checkpoint_folder, model, last_global_step, lr_scheduler, accelerator):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "last_global_step": last_global_step,
    }

    accelerator.print("==> saving model and optimizer <==")
    model.save_checkpoint(checkpoint_folder, last_global_step, checkpoint_state_dict)

    accelerator.print("==> saving lr scheduler <==")
    accelerator.save(lr_scheduler.state_dict(), os.path.join(checkpoint_folder, str(last_global_step), "scheduler.pt"))

    print(f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={last_global_step}")
    return

def resume_model_and_optimizer(model, load_dir, tag):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag = tag, load_optimizer_states = True)
    
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict

    return last_global_step

def checkpoint_model(accelerator, model, tokenizer, output_ckpt_dir, last_global_step):    
    '''
    Utility fuction for only checkpointing the model dictionary (i.e., only model parameters)
    '''
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    accelerator.print("checkpointing model state dict at {}".format(ckpt_path))
    unwrapped_model = accelerator.unwrap_model(model)
    # TODO: currently, there is a small bug that saves a full checkpoint data for each shard when enable zero1 and 2. 
    # See https://github.com/microsoft/DeepSpeed/issues/3303. solution: waiting upgrade of accelerate and deepspeed
    unwrapped_model.save_pretrained(
        ckpt_path, 
        is_main_process = accelerator.is_main_process, 
        save_function = accelerator.save,
        state_dict = accelerator.get_state_dict(model),
        max_shard_size = "100GB"
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_path)
    
    return

def train(opt):
    set_seed(opt.seed)

    writer = SummaryWriter(opt.tensorboard_log_dir)
    accelerator = Accelerator()
    print("accelerator.is_main_process:", accelerator.is_main_process)
    print("accelerator.device:", accelerator.device)

    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    accelerator.print(opt)
    accelerator.print("tokens per batch:", total_batch_size * opt.block_size)
    accelerator.print("sequences per batch:", total_batch_size)
    accelerator.print("using LLM from:", opt.pretrained_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    model.gradient_checkpointing_enable()

    if opt.mode == "pt":
        dataset = PretrainDataset(opt.pt_data_dir, opt.block_size)
    elif opt.mode == "sft":
        dataset = SFTSQLGenerationDataset(opt.text2sql_data_dir, tokenizer, opt.block_size, "train", opt.table_num, opt.column_num, None)
    else:
        raise ValueError("opt.mode should be in [pt, sft].")
    dataloader = DataLoader(dataset, batch_size = opt.per_device_train_batch_size, shuffle = True, drop_last = True)

    num_total_batches = math.ceil(opt.epochs * math.ceil(len(dataset) / total_batch_size)) # number of total batches
    optimizer = AdamW(model.parameters(), lr = opt.lr, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.1)

    num_warm_up_batches = max(int(num_total_batches * opt.warmup_ratio), 1)
    accelerator.print("num_warm_up_batches:", num_warm_up_batches)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer = optimizer, 
        warmup_epochs = num_warm_up_batches * accelerator.num_processes,
        max_epochs = num_total_batches * accelerator.num_processes, 
        warmup_start_lr = 0.0, 
        eta_min = 0.1 * opt.lr
    )

    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(optimizer, model, dataloader, lr_scheduler)
    print(type(optimizer))
    print(type(model))
    print(type(dataloader))
    print(type(lr_scheduler))

    accumulation_loss = 0
    global_completed_steps = 0
    model.train()

    # resume pre-training if opt.resume_from_checkpoint is not None
    if opt.mode == "pt" and opt.resume_from_checkpoint:
        # resume model and optimizer states
        global_completed_steps = resume_model_and_optimizer(model, opt.resume_from_checkpoint, opt.resume_tag)
        
        resume_epoch = global_completed_steps * accelerator.gradient_accumulation_steps // len(dataloader)
        resume_batch_idx = global_completed_steps * accelerator.gradient_accumulation_steps % len(dataloader)

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print("resume training from {}".format(os.path.join(opt.resume_from_checkpoint, opt.resume_tag)))

        # resume lr scheduler
        lr_scheduler.load_state_dict(torch.load(os.path.join(opt.resume_from_checkpoint, opt.resume_tag, "scheduler.pt")))
        accelerator.print("lr scheduler state dict:", lr_scheduler.state_dict())

    st = time.time()
    for epoch in range(opt.epochs):
        if opt.mode == "pt" and opt.resume_from_checkpoint and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            continue
        accelerator.print("Start training epoch:", epoch+1)
        for batch_idx, batch in enumerate(dataloader):
            # if batch["attention_mask"].all():
            # accelerator.print("\n----------------\n".join(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = False)))
            # accelerator.print(batch["input_ids"])
            # accelerator.print(batch["attention_mask"])
            # accelerator.print(batch["labels"])
            # accelerator.print(batch["input_ids"].shape)

            if opt.mode == "pt" and opt.resume_from_checkpoint and resume_batch_idx > batch_idx:
                accelerator.print("skip {}-th batch".format(batch_idx))
                continue
            
            # `accelerator.accumulate(model)` aims to set right `sync_gradients` state based on the recorded training steps
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accumulation_loss += loss.detach().float()
                # when deepspeed is enabled, `accelerator.backward(loss)` is doing optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` examples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                accelerator.print("GPU 0, step {}, loss {}".format(global_completed_steps, accumulation_loss / accelerator.gradient_accumulation_steps))
                accelerator.print("GPU 0, step {}, lr state dict:".format(global_completed_steps), lr_scheduler.state_dict())
                accelerator.print(time.time()-st)
                st = time.time()

                writer.add_scalar(
                    'train-loss/gpu-{}'.format(accelerator.process_index), 
                    accumulation_loss / accelerator.gradient_accumulation_steps, 
                    global_completed_steps
                )
                writer.add_scalar(
                    'learning-rate/gpu-{}'.format(accelerator.process_index), 
                    lr_scheduler.get_last_lr()[0], 
                    global_completed_steps
                )
                # reset accumulation_loss to 0
                accumulation_loss = 0

                # save checkpoints for each checkpointing_steps total batch size
                if global_completed_steps % opt.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)
                    if opt.save_all_states:
                        checkpoint_model_optimizer_scheduler(opt.output_ckpt_dir, model, global_completed_steps, lr_scheduler, accelerator)

        # if opt.mode == "pt" or (opt.mode == "sft" and (epoch+1)%2 == 0):
        accelerator.print("in the end of an epoch, save a checkpoint")
        accelerator.wait_for_everyone()
        checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)
        if opt.save_all_states:
            checkpoint_model_optimizer_scheduler(opt.output_ckpt_dir, model, global_completed_steps, lr_scheduler, accelerator)

if __name__ == "__main__":
    opt = parse_option()
    train(opt)