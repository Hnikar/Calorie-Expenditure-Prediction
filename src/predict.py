import torch
import pandas as pd
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from src.model import NeuralNetwork
from src.workout_dataset import load_data

@hydra.main(version_base=None, config_path="config", config_name="config")
def predict(cfg: DictConfig):
    print("Calculating normalization parameters based on train.csv...")
    train_x, _ = load_data(cfg.data.train)
    x_min, _ = train_x.min(dim=0, keepdim=True)
    x_max, _ = train_x.max(dim=0, keepdim=True)

    print("Loading and processing test.csv...")
    test_df = pd.read_csv(cfg.data.test)
    ids = test_df['id'] 
    
    test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0}).astype(float)
    features = test_df.drop(columns=['id'])
    
    x_test = torch.tensor(features.values, dtype=torch.float32)

    x_test = (x_test - x_min) / (x_max - x_min + 1e-8)

    model_path = "/app/outputs/2025-12-18/17-34-42/best_model.pth" 
    
    if not os.path.exists(model_path):
        import glob
        list_of_files = glob.glob('/app/outputs/**/best_model.pth', recursive=True)
        if list_of_files:
            model_path = max(list_of_files, key=os.path.getctime)
            print(f"Found latest checkpoint: {model_path}")
        else:
            print("Error: best_model.pth not found. Please train the model first!")
            return

    model = NeuralNetwork(
        input_dim=x_test.shape[1],
        hidden_dim=cfg.model.hidden_dim,
        dropout_rate=0.0
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Generating predictions...")
    with torch.no_grad():
        predictions = model(x_test).numpy().flatten()

    submission = pd.DataFrame({'id': ids, 'Calories': predictions})
    submission.to_csv('submission.csv', index=False)
    print(f"File submission.csv saved in {os.getcwd()}")

if __name__ == "__main__":
    predict()