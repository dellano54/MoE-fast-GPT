from datasets import load_dataset
import json
import torch
import math
from tokenizer import Tokenizer
from modelling import GPT

with open("hparams.json", "r") as f:
    HPARAM = json.loads(f.read())


if HPARAM['early_stopping'] == 0:
    HPARAM['early_stopping'] = None

if HPARAM['dtype'] == 'fp16':
    dtype = torch.float16

elif HPARAM['dtype'] == 'bf16':
    dtype = torch.bfloat16

else:
    raise ValueError("dtype can only be fp16 or bf16")


# load the tokenizer
TOKENIZER = Tokenizer(HPARAM['tokenizer'], HPARAM['max_length'])


# fineweb dataset to pretrain our LM
DATASET = load_dataset(HPARAM['dataset'], name=HPARAM['split'], split="train", streaming=True)


#collate function for trainLoader
def collate_fn(batch):
    batch = [b['text'] for b in batch] 
    input_ids = TOKENIZER(batch)
    
    return input_ids['input_ids'], input_ids['attention_mask']


#training function

total_losses = []

scaler = torch.amp.GradScaler()

def train(model, optimizer, scheduler, trainData, device):
    model.to(device)
    model.train()

    running_loss = 0
    prev_idx = 0

    for epoch in range( HPARAM['epochs'] ):
        for step, (input_ids, attention_mask) in enumerate(trainData):
            input_ids, attention_mask = input_ids.to(HPARAM['device']), attention_mask.to(HPARAM['device'])

            optimizer.zero_grad()

            with torch.amp.autocast(HPARAM['device'], dtype=dtype):
                loss, _, moe_loss = model(
                    input_ids,
                    attention_mask
                )
            
            loss = loss.mean()
            loss /= HPARAM['gradient_accumulation']
            scaler.scale(loss).backward()
            

            if step%HPARAM['gradient_accumulation'] == 0:
              scaler.step(optimizer)
              scaler.update()

              if scheduler is not None:
                  scheduler.step()

            total_losses.append(loss.item() * HPARAM['gradient_accumulation'])
            running_loss += total_losses[-1]

            print(f"\repoch: {epoch} - step: {step} - loss: {total_losses[-1]} -moe : {moe_loss.mean().item()}- lr: {optimizer.param_groups[0]['lr']}", end="")

            if (step%HPARAM['log_steps'] == 0) and (step != 0):
                print(f"\navg loss: {running_loss / HPARAM['log_steps']}")
                running_loss = 0

                torch.save(
                    {'model': model.state_dict(),
                    "loss": sum(total_losses[ prev_idx : prev_idx+HPARAM['log_steps'] ]) / HPARAM['log_steps'],
                    "step": step,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "total_loss": total_losses},

                    HPARAM['save_path']
                )

                prev_idx += HPARAM['log_steps']

            if step == HPARAM['early_stopping']:
                print(f"\navg loss: {running_loss / HPARAM['log_steps']}")

                torch.save(
                    {'model': model.state_dict(),
                    "loss": sum(total_losses[ prev_idx : prev_idx+HPARAM['log_steps'] ]) / HPARAM['log_steps'],
                    "step": step,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "total_loss": total_losses},

                    HPARAM['save_path']
                )

                return



# cosine lr schedyler
class GPTLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, lr_decay_iters: int,
                 base_lr: float, min_lr: float, last_epoch: int = -1):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        lrs = []
        for _ in self.optimizer.param_groups:
            # warmup phase – linearly increase LR
            if it < self.warmup_iters:
                lr = self.base_lr * (it + 1) / (self.warmup_iters + 1)
            # post-decay – clamp to min_lr
            elif it > self.lr_decay_iters:
                lr = self.min_lr
            else:
                # cosine decay between warmup and decay_iters
                decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  #  cosine interpolation
                lr = self.min_lr + coeff * (self.base_lr - self.min_lr)
            lrs.append(lr)
        return lrs



#ignore layerNorm and embedding from weight_decay 
def get_param_groups(model, weight_decay=0.01):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    decay = set()
    no_decay = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            # full parameter name for debugging, optional
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if isinstance(module, (torch.nn.RMSNorm, torch.nn.Embedding)):
                no_decay.add(param)
            elif param_name.endswith("bias"):
                no_decay.add(param)
            else:
                decay.add(param)

    # Remove any overlaps just in case
    decay = decay - no_decay

    return [
        {"params": list(decay), "weight_decay": weight_decay},
        {"params": list(no_decay), "weight_decay": 0.0},
    ]


        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GPT(
    vocab_size=len(TOKENIZER),
    embed_dim=512,
    n_heads=8,
    feed_forward=2048,
    num_layers=4,
    max_length=HPARAM['max_length'],
    num_experts=8,
    k=2,
    dropout=0.1,
    pad_token_id=TOKENIZER.pad_token_id,
    alpha= 0.3
)



print(f"num of parameters: {count_parameters(model)}")

optimizer = torch.optim.AdamW(
    get_param_groups(model, HPARAM['weight_decay']),
    lr=HPARAM['lr'],
    weight_decay=0.0
)


scheduler = GPTLRScheduler(
    optimizer,
    HPARAM['warmup_steps'],
    HPARAM['train_steps'],
    HPARAM['lr'],
    HPARAM['min_lr']
)


trainDataLoader = torch.utils.data.DataLoader(
    DATASET,
    batch_size=HPARAM['batch_size'],
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

model = torch.nn.DataParallel(model)
print(model)
train(
    model,
    optimizer,
    scheduler,
    trainDataLoader,
    HPARAM['device']
)