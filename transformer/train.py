import torch 
import torch.nn as nn
from model import build_transformer
from config import get_config, get_weights_filename
from dataset import data_loader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from dataset import EOS_TOKEN_ID, PAD_TOKEN_ID
import warnings

# model to get transformer model details
def get_model(config, src_vocab_size: int, tgt_vocab_size: int):
    model =  build_transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        config["context_length"],
        config["context_length"],
        config["feature_dims"]
    )

    return model 


# Training and validation of Transformer model

# Definition of Training loop 
def build_training(config):
    # Define cuda device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device running on {device}')

    # Create weight's folder to store trained weights 
    Path(config["model_folder"]).mkdir(parents= True, exist_ok= True)

    # Load datasets and model in this training function 
    train_dataloader, validation_dataloader, tokenizer = data_loader(config)
    # Increasing size of tokenization (50257) to (50260) (SOS, PAD, EOS) tokens
    vocab_size = EOS_TOKEN_ID + 1 # 50260 + 1
    model = get_model(config, vocab_size, vocab_size).to(device)

    # Building tensor board for loss func visualization 
    writer = SummaryWriter(config["experiment_name"])

    # Optimizer used: Adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr= config["learning_rate"], eps= 1e-9)

    # Running optimization using preload of weights 
    initial_epoch = 0
    global_step = 0
    
    if config["preload"]:
        model_filename = get_weights_filename(config, config["preload"])
        print(f"Model running on {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    
    # loss function: CrossEntropyLoss 
    lossfn = nn.CrossEntropyLoss(ignore_index= PAD_TOKEN_ID, label_smoothing= 0.1).to(device)

    # Training loop 
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc= f"processing epoch: {epoch:02d}")
        
        # For every batch (encoder_input, decoder_input, src_mask, tgt_mask)
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   # (B, 1, context_length)
            decoder_input = batch['decoder_input'].to(device)   # (B, 1, context_length)
            src_mask = batch['src_mask'].to(device) # (B, 1, context_length)
            tgt_mask = batch['tgt_mask'].to(device) # (B, 1, context_length, context_length)

            # Run this tensors into the transformer model (encoder, decoder and projection layer)
            encoder_output = model.encoder_layer(encoder_input, src_mask)
            decoder_output = model.decoder_layer(decoder_input, encoder_output, src_mask, tgt_mask)
            projection = model.projection_layer(decoder_output) #(B, context_length, vocab_size)

            # Label: comparable value tensor matrix 
            label = batch['label'].to(device) # (B, vocab_size)

            # Calculating loss by making prejection layer dims w.r.t label dims 
            loss = lossfn(projection.view(-1, vocab_size), label.view(-1))
            batch_iterator.set_postfix({
                f"loss": f"{loss.item():6.3f}"
            })

            # Adding loss to tensorboard 
            writer.add_scalar('Train loss', loss.item(), global_step)
            writer.flush()

            # Back-propagate the loss to minimize loss and update weights 
            loss.backward()

            # Update each layer weight 
            optimizer.step()
            optimizer.zero_grad()
        
        # Saving the weights values after each epoch 
        model_filename = get_weights_filename(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "global_step": global_step

            }, model_filename
        )


if __name__ == "__main__":
    # ignore warnings 
    warnings.filterwarnings('ignore')
    config = get_config()
    build_training(config)

