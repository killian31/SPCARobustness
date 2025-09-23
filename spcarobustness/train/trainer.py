from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def train_pipeline_model(
    model: nn.Module,
    X_train,
    y_train,
    device,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        if verbose:
            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
    return model, optimizer, criterion
