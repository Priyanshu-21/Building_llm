import torch 
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import tiktoken

# Definition of dataloader and datasets with tokenization (gpt-2) scheme
# END_OF_TEXT = 50257
PAD_TOKEN_ID = 50258
SOS_TOKEN_ID = 50259
EOS_TOKEN_ID = 50260

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer: tiktoken.Encoding, context_length: int) -> None: 
        super().__init__() # call to dataset class to load init and other methods 
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length
    
    # Length of dataset
    def __len__(self):
        return len(self.dataset)
    
    # Lookup dataset table for each token 
    def __getitem__(self, idx):
        # Load datasets into pairs of source and target language
        pair = self.dataset[idx]
        src_text = pair["english_sentence"]
        tgt_text = pair["hindi_sentence"]

        # Tokenizing each src_text and tgt_text 
        src_tokens = self.tokenizer.encode(src_text, allowed_special= {"<|endoftext|>"}) 
        tgt_tokens = self.tokenizer.encode(tgt_text, allowed_special= {"<|endoftext|>"})

        # Introduction of padding in source and target tokens 
        src_pad_counts = self.context_length - len(src_tokens) - 2 # 1 for SOS, 1 for EOS
        tgt_pad_counts = self.context_length - len(tgt_tokens) - 1 # 1 for either SOS, EOS

        # src, tgt padding counts should be greater than 0
        if src_pad_counts < 0 or tgt_pad_counts < 0:
            raise ValueError(
                f"Index {idx} exceeds context_length={self.context_length}."
                f"src_tokens={len(src_tokens)}, tgt_tokens={len(tgt_tokens)}"
            )
        
        # Now with this make encoder and decoder tokenized tensor to feed to positional encoding layer 
        encoder_input = torch.tensor(
            [SOS_TOKEN_ID] + src_tokens + [EOS_TOKEN_ID] + [PAD_TOKEN_ID] * src_pad_counts, 
            dtype= torch.int64
        )

        decoder_input = torch.tensor(
            [SOS_TOKEN_ID] + tgt_tokens + [PAD_TOKEN_ID] * tgt_pad_counts,
            dtype= torch.int64
        )

        label = torch.tensor(
            tgt_tokens + [EOS_TOKEN_ID] + [PAD_TOKEN_ID] * tgt_pad_counts,
            dtype= torch.int64
        )
        
        # checking if all calculation is equal to context_length or not
        # each row of encoder, decoder and label == context_length
        assert encoder_input.size(0) == self.context_length
        assert decoder_input.size(0) == self.context_length
        assert label.size(0) == self.context_length

        # Masking, self-attention mask: - encoder; Casual-attention mask: - decoder 
        # mask dim (1, 1, context_length)
        src_mask = (encoder_input != PAD_TOKEN_ID).unsqueeze(0).unsqueeze(0).bool()
        tgt_padding_mask = (decoder_input != PAD_TOKEN_ID).unsqueeze(0).unsqueeze(0).bool()
        tgt_casual_mask = casual_mask(self.context_length)
        # Actual casual mask in decoder 
        tgt_mask = tgt_padding_mask & tgt_casual_mask

        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input, 
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "label": label,
            "src_text": src_text, 
            "tgt_text": tgt_text
        }


def casual_mask(size: int) -> torch.Tensor:
    # Upper triangular matrix with all value above diagonal == 1 is set to 0
    mask = torch.triu((torch.ones(1, size, size)), diagonal= 1)
    return mask == 0


# Dataloader function: training, validation datasets 
def data_loader(config: dict): 
    # Defining tokenizer used: gpt2 (50257 tokens --> vocab size)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Loading dataset 
    raw_data = load_dataset(config["datasource"], split= "train")

    # Filtering datasets exceeding context_length limit 
    context_length = config["context_length"]
    max_src_length = context_length - 2 # SOS and EOS
    max_tgt_legth = context_length - 1 # SOS or EOS

    # Calculating valid lenght per source and target 
    def is_valid_length(example):
        src_len = len(tokenizer.encode(example["english_sentence"]))
        tgt_len = len(tokenizer.encode(example["hindi_sentence"]))

        return src_len <= max_src_length and tgt_len <= max_tgt_legth
    
    # Filtered dataset by context length 
    filtered_dataset = raw_data.filter(is_valid_length, desc="Filtering by context length")

    # Split this filtered datasets into training, validation sets 
    val_split = config.get("val_split", 0.1)
    val_size = int(val_split * len(filtered_dataset))
    train_size = int(len(filtered_dataset) - val_size)

    # Randomly splitting filtered_dataset into 90% -> training, 10% into validation 
    train_raw, val_raw = random_split(filtered_dataset, [train_size, val_size]) # type: ignore

    # Getting training and validation datasets from datasets class impl 
    train_datasets = BilingualDataset(train_raw, tokenizer, context_length)
    val_datasets = BilingualDataset(val_raw, tokenizer, context_length)

    num_workers = config.get("num_workers", 0) # How many gpu's running parallel 
    
    # Dataloader for each batch of training and validation set 
    train_dataloader = DataLoader(
        train_datasets, 
        batch_size= config["batch_size"],
        shuffle= True,
        num_workers= num_workers,
        pin_memory= True, 
        drop_last= False, 
    )

    validation_dataloader = DataLoader(
        val_datasets,
        batch_size= 1,
        shuffle= False,
        num_workers= num_workers, 
        pin_memory= True,
        drop_last= False
    )

    return train_dataloader, validation_dataloader, tokenizer

