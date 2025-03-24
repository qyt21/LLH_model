import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

X_base = pd.read_csv('reconstructed_curve.csv', header=None).values  
X_35 = pd.read_csv('features.csv', header=None).values  

features_CT = pd.read_csv('features_mink.csv', header=None)
features_CT.columns = ['M0', 'M1', 'M2', 'M3']
y_target = features_CT.values

def objective_function(params, data, sigma_target=1):
    residuals = []
    for _, row in data.iterrows():
        M0, M1, M2, M3 = row['M0'], row['M1'], row['M2'], row['M3']
        sigma_log_predicted = params[0]*M0 + params[1]*M1 + params[2]*M2 + params[3]*M3
        residuals.append((sigma_log_predicted - np.log(sigma_target)) ** 2)
    return np.sum(residuals)

features_CT_scaled = features_CT / features_CT.max().values
initial_guess = [1, 1, 1, 1]
bounds = [(-10, 10)] * 4
result = minimize(objective_function, initial_guess, args=(features_CT_scaled,), bounds=bounds)
alpha1_opt, alpha2_opt, alpha3_opt, alpha4_opt = result.x

print("Optimized coefficients:")
print("α₁ =", alpha1_opt)
print("α₂ =", alpha2_opt)
print("α₃ =", alpha3_opt)
print("α₄ =", alpha4_opt)

def calculate_sigma(alpha1, alpha2, alpha3, alpha4, M0, M1, M2, M3, sigma_ref=1):
    return sigma_ref * np.exp(alpha1*M0 + alpha2*M1 + alpha3*M2 + alpha4*M3)

M0 = features_CT['M0'].values
M1 = features_CT['M1'].values
M2 = features_CT['M2'].values
M3 = features_CT['M3'].values
sigma = calculate_sigma(alpha1_opt, alpha2_opt, alpha3_opt, alpha4_opt, M0, M1, M2, M3)

sigma_normalized = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma))

# construct final input features
X_full = np.hstack((X_base, X_35, sigma_normalized.reshape(-1, 1)))
print("Final input shape:", X_full.shape)

# split and scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_full, y_target, test_size=0.2, random_state=42
)

X_train_scaled = scaler_X.fit_transform(X_train_split)
X_test_scaled = scaler_X.transform(X_test_split)
y_train_scaled = scaler_y.fit_transform(y_train_split)
y_test_scaled = scaler_y.transform(y_test_split)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class DNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2_res = nn.Linear(512, 512)
        self.bn2_res = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.25)
        self.act3 = nn.ELU()
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.act4 = nn.LeakyReLU(negative_slope=0.01)
        self.out = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.act1(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        out_base = self.act2(self.bn2(self.fc2(out)))
        out = self.dropout2(out_base)       
        res = self.bn2_res(self.fc2_res(out))
        out = out + res       
        out = self.act3(self.bn3(self.fc3(out)))
        out = self.dropout3(out)    
        out = self.act4(self.bn4(self.fc4(out)))        
        output = self.out(out)
        return output

input_size = X_train_scaled.shape[1]  
output_size = y_target.shape[1]        
model = DNNModel(input_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

epochs = 500
best_val_loss = float('inf')
patience = 50
counter = 0
best_model = None

for epoch in range(epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_train_loss = np.mean(train_losses)
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_pred = model(X_val)
            loss_val = criterion(val_pred, y_val)
            val_losses.append(loss_val.item())
    avg_val_loss = np.mean(val_losses)
    
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered at epoch:", epoch+1)
            break
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

if best_model is not None:
    model.load_state_dict(best_model)


model.eval()
with torch.no_grad():
    predictions_test = model(X_test_tensor.to(device))
    predictions_test = predictions_test.cpu().numpy()
    y_test_inv = scaler_y.inverse_transform(y_test_scaled)
    predictions_inv = scaler_y.inverse_transform(predictions_test)

mae_test = mean_absolute_error(y_test_inv, predictions_inv)
mse_test = mean_squared_error(y_test_inv, predictions_inv)
r2_test = r2_score(y_test_inv, predictions_inv)

print("Test Set Metrics:")
print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"R-squared (R^2): {r2_test:.4f}")


features = ['M0', 'M1', 'M2', 'M3']

# plot each feature
for i, feature in enumerate(features):
    plt.figure()
    plt.scatter(y_test_inv[:, i], predictions_inv[:, i], alpha=0.6, label="Test Data")
    plt.plot([y_test_inv[:, i].min(), y_test_inv[:, i].max()],
             [y_test_inv[:, i].min(), y_test_inv[:, i].max()], 'r--', label="Ideal Fit")
    plt.xlabel(f'Actual {feature}')
    plt.ylabel(f'Predicted {feature}')
    plt.title(f'Predicted vs Actual for {feature}')
    plt.legend()
    plt.show()

# save predictions and actuals
df_m0 = pd.DataFrame({
    'Predicted_M0': predictions_inv[:, 0],
    'Actual_M0':    y_test_inv[:, 0]
})
df_m0.to_csv('predicted_M0.csv', index=False)

df_m1 = pd.DataFrame({
    'Predicted_M1': predictions_inv[:, 1],
    'Actual_M1':    y_test_inv[:, 1]
})
df_m1.to_csv('predicted_M1.csv', index=False)

df_m2 = pd.DataFrame({
    'Predicted_M2': predictions_inv[:, 2],
    'Actual_M2':    y_test_inv[:, 2]
})
df_m2.to_csv('predicted_M2.csv', index=False)

df_m3 = pd.DataFrame({
    'Predicted_M3': predictions_inv[:, 3],
    'Actual_M3':    y_test_inv[:, 3]
})
df_m3.to_csv('predicted_M3.csv', index=False)

print("CSV files with predicted (first column) and actual (second column) have been saved for M0, M1, M2, M3.")
