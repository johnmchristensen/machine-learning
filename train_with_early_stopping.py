import torch

from evaluate_tm import evaluate_tm
from get_device import get_device
from history import History

def train_with_early_stopping(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs,
                              patience=10, checkpoint_path=None, scheduler=None):
    def best_callback():
        torch.save(model.state_dict(), checkpoint_path)

    checkpoint_path = checkpoint_path or "my_checkpoint.pt"
    history = History(n_epochs, patience)
    for epoch in range(n_epochs):
        metric.reset()
        model.train()
        with history.start():
            total_loss = __train_batch(model, train_loader, loss_fn, optimizer, metric)
            train_metric = metric.compute().item()
            valid_metric = evaluate_tm(model, valid_loader, metric).item()

        history.append(train_loss=total_loss / len(train_loader), train_metric=train_metric, valid_metric=valid_metric,
                       best_callback=best_callback)
        history.print()

        if scheduler is not None:
            scheduler.step()
        if history.is_out_of_patience():
            print("Early stopping!")
            break
    model.load_state_dict(torch.load(checkpoint_path))
    return history

def __train_batch(model, loader, loss_fn, optimizer, metric):
    device = get_device()
    total_loss = 0.0
    for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
    return total_loss