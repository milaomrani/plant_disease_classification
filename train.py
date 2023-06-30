import torch
from dataLoader import get_train_dataloader, get_test_dataloader
from model import TinyVgg, ImageNetModel
from options import parse_arguments
from utils import save_model
from timeit import default_timer as timer 



def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, y_pred_class = torch.max(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y)
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print(f'Epoch: {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f}')

        # Evaluate the model on the test set
        test_loss, test_acc = evaluate(model, test_dataloader, loss_fn, device)
        print(f'Test Loss: {test_loss:.4f} | '
              f'Test Acc: {test_acc:.4f}')

        
        # save_model(model=model,
        #                target_dir=args.model_path,
        #                model_name= f"models_{epoch}")
        
        torch.save(model.state_dict(), args.model_path)

        print('---------------------------------------')

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            _, y_pred_class = torch.max(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

if __name__ == '__main__':
    
    start_time = timer()
    args = parse_arguments()

    device = torch.device(args.device)

    train_dataloader = get_train_dataloader(args.train_dir, args.batch_size, 1)
    test_dataloader = get_test_dataloader(args.test_dir, args.batch_size, 1)

    if args.model == "ImageNet":
        model = ImageNetModel(output_shape=len(train_dataloader.dataset.classes)).to(device)
    else:
        model = TinyVgg(input_shape=3, hidden_units=args.hidden_units, output_shape=len(train_dataloader.dataset.classes)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train(model, train_dataloader, test_dataloader, optimizer, loss_fn, args.num_epochs, device)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

