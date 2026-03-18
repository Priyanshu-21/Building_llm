from pathlib import Path

def get_config():
    return {
        "datasource": "Aarif1430/english-to-hindi",
        "batch_size": 8,
        "learning_rate": 10e-4,
        "context_length": 256,
        "feature_dims": 512,
        "num_epochs": 20,
        "src_language": "en",
        "tgt_language": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel"  
    }

def get_weights_filename(config, epoch: str): 
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    # Save model with correct format
    return str(Path(".") / model_folder / model_filename)