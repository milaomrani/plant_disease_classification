import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training options")
    parser.add_argument("--train_dir", type=str, default="E:/plant_detection/DATASET_KAGGLE/train", help="Path to the train directory")
    parser.add_argument("--test_dir", type=str, default="E:/plant_detection/DATASET_KAGGLE/test", help="Path to the test directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()
    return args
