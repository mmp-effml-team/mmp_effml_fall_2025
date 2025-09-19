import torch
from torch import nn
import numpy as np
import random
from tqdm.auto import tqdm
from typing import Callable

from argparse import ArgumentParser


from torch.utils.data import DataLoader
import data_utils
from positional_encoding import PositionalEncoding
from functools import partial
from transformers import AutoTokenizer

import amp
import utils


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class GPT2LikeModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1023, num_heads=8, max_length=data_utils.MAX_LENGTH):
        super().__init__()
        self.num_heads = 8
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_length)
        self.hidden_projector = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_projector2 = nn.Linear(embedding_dim, hidden_dim)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.output_linear = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask):
        x = x.transpose(0, 1) # as we don't use batch first
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.hidden_projector(x)
        y = self.hidden_projector2(x)
        if attention_mask is not None:
            attention_mask = attention_mask.to(x.device)
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(torch.float32).masked_fill_(attention_mask.logical_not(), float("-inf"))
                attention_mask = attention_mask.to(x.dtype)

        out = self.decoder(tgt=x, memory=x, tgt_mask=attention_mask, memory_mask=attention_mask)
        out = self.output_linear(out)
        return out.transpose(0, 1)


def get_gpt2_model(vocab_size) -> torch.nn.Module:
    return GPT2LikeModel(vocab_size)


def get_dataloader(dataloader_type, batch_size, path, k):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataloader = None
    if dataloader_type == 'base':
        ds = data_utils.BaseDataset(path, tokenizer)
        collate = partial(data_utils.base_collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    if dataloader_type == 'standard':
        ### YOUR CODE HERE
        # prepare dataloader with StandardDataset and collate_fn
        pass
    if dataloader_type == 'balanced':
        ### YOUR CODE HERE
        # prepare dataloader with StandardDataset collate_fn and BalancedBatchSampler
        pass
    if dataloader_type == 'sequenced':
        ### YOUR CODE HERE
        # prepare SequencedDataset
        pass
    return dataloader


def train_epoch(
    train_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    metric: Callable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training_kind: str,
    scaler: None | amp.StaticGradScaler | amp.DynamicGradScaler,
) -> None:
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for i, data in pbar:
        tokens, tokens_lens, attention_mask = data['tokens'].to(device), data['lengthes'], data['attention_mask']
        
        # obtain loss value depending on training_kind 
        ### YOUR CODE HERE
        
        optimizer.zero_grad(set_to_none=False)
        if training_kind == 'fp16' or training_kind == 'fp32':
            # compute grads without scaling
            ### YOUR CODE HERE
            pass
        else:
            # compute grads with scaling
            ### YOUR CODE HERE
            pass

        accuracy = metric(outputs, tokens, tokens_lens)

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")

def parse_args():
    parser = ArgumentParser(description="Training Acceleration Task")
    parser.add_argument("--training_kind", choices=["fp32", "fp16", "fp16_static_scaler", "fp16_dynamic_scaler"], default="fp32", help="Training kind (dtype and scaler)")
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of epochs')

    parser.add_argument("--dataloader", choices=["base", "standard", "balanced", "sequenced"], default="base", help="Dataloader type")
    parser.add_argument("--path", type=str, default='data/validation-00000-of-00001.txt', help='Dataset path')
    parser.add_argument("--k", type=int, default=20, help='k for balanced')

    return parser.parse_args()


def train():
    set_global_seed(42)
    args = parse_args()
    device = torch.device("cuda:0")
    dataloader = get_dataloader(args.dataloader, args.batch_size, args.path, args.k)
    model = get_gpt2_model(dataloader.dataset.tokenizer.vocab_size).to(device)
    criterion = utils.LMCrossEntropyLoss()
    metric = utils.LMAccuracy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = None
    if args.kind == 'fp16_static_scaler':
        # initialize static scaler
        ### YOUR CODE HERE
        pass
    elif args.kind == 'fp16_dynamic_scaler':
        # initialize dynamic scaler
        ### YOUR CODE HERE
        pass
    num_epochs = args.num_epochs
    for epoch in range(0, num_epochs):
        train_epoch(
            train_loader=dataloader, 
            model=model, 
            criterion=criterion, 
            metric=metric,
            optimizer=optimizer, 
            device=device, 
            kind=args.kind, 
            scaler=scaler
        )


if __name__ == '__main__':
    train()
