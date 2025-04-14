import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class OptimizedDNN(nn.Module):
    def __init__(self, num_features, num_outputs):
        super(OptimizedDNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        self.fc_out = nn.Linear(128, num_outputs) 
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln4(self.fc4(x)))
        x = self.fc_out(x)

        return x

def train_optimized_dnn(
    masked_file, full_file, output_pred_file,
    epochs=500, batch_size=32, learning_rate=0.0003, weight_decay=1e-4
):
    
    masked_df = pd.read_csv(masked_file)
    full_df   = pd.read_csv(full_file)
    full_columns = full_df.columns
    
    imputer = SimpleImputer(strategy="mean")
    masked_imputed = imputer.fit_transform(masked_df.values)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        masked_imputed, full_df.values, test_size=0.2, random_state=42
    )

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_np)
    y_train_scaled = scaler_y.fit_transform(y_train_np)
    X_test_scaled = scaler_x.transform(X_test_np)
    y_test_scaled = scaler_y.transform(y_test_np)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    num_features = X_train.shape[1]
    num_outputs  = y_train.shape[1]
    model = OptimizedDNN(num_features, num_outputs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print("NaN loss detected, stopping training.")
                return

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        masked_imputed_scaled = scaler_x.transform(masked_imputed)
        masked_imputed_tensor = torch.tensor(masked_imputed_scaled, dtype=torch.float32).to(device)
        y_pred_scaled = model(masked_imputed_tensor).cpu().detach().numpy()
        y_test_pred_scaled = model(X_test).cpu().detach().numpy()

    y_pred      = scaler_y.inverse_transform(y_pred_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test.cpu().numpy())

    predicted_df = pd.DataFrame(y_pred, columns=full_columns)
    predicted_df.to_csv(output_pred_file, index=False)

    rmse = np.sqrt(mean_squared_error(y_test_orig.flatten(), y_test_pred.flatten()))
    r2   = r2_score(y_test_orig.flatten(), y_test_pred.flatten())
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2 Score: {r2:.4f}")


    plt.figure(figsize=(8, 5))
    plt.plot(y_test_orig[0], label="True Curve", color="blue")
    plt.plot(y_test_pred[0], label="Predicted Curve", linestyle="dashed", color="red")
    plt.xlabel("Point Index")
    plt.ylabel("Value")
    plt.title("Optimized Fully Connected DNN (PyTorch) on GPU: True vs. Predicted Curve")
    plt.legend()
    plt.show()


# Load your file before run
# for example: train_optimized_dnn(masked_csv, target_csv, predicted_csv)
