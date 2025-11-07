"""
Machine Learning Models Module
Neural network models with custom Sharpe ratio loss function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


class SharpeLoss(nn.Module):
    """
    Custom loss function that optimizes for Sharpe ratio.
    Maximizes risk-adjusted returns instead of just minimizing prediction error.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe loss.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        super(SharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss.
        
        Args:
            predictions: Model predictions (position signals)
            targets: Actual returns
            
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Calculate strategy returns
        strategy_returns = predictions.squeeze() * targets.squeeze()
        
        # Calculate Sharpe ratio
        mean_return = torch.mean(strategy_returns)
        std_return = torch.std(strategy_returns) + 1e-8  # Avoid division by zero
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Return negative Sharpe (we want to minimize loss, maximize Sharpe)
        return -sharpe


class TradingDataset(Dataset):
    """PyTorch dataset for trading data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class TradingNN(nn.Module):
    """
    Neural network for trading signal generation.
    Outputs continuous values representing position strength.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(TradingNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (single value for position signal)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Output between -1 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class ModelTrainer:
    """
    Trains trading models with custom loss functions.
    """
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module,
                 learning_rate: float = 0.001, device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            loss_fn: Loss function
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_sharpes = []
        self.val_sharpes = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        sharpe = self._calculate_sharpe(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, sharpe
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self.loss_fn(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        sharpe = self._calculate_sharpe(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, sharpe
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, early_stopping_patience: int = 20) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining on {self.device}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_sharpe = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_sharpes.append(train_sharpe)
            
            # Validate
            val_loss, val_sharpe = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_sharpes.append(val_sharpe)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Sharpe: {train_sharpe:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}, Val Sharpe:   {val_sharpe:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        print("=" * 60)
        print("Training complete!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_sharpes': self.train_sharpes,
            'val_sharpes': self.val_sharpes
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for features, _ in data_loader:
                features = features.to(self.device)
                preds = self.model(features)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    @staticmethod
    def _calculate_sharpe(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        strategy_returns = predictions.flatten() * targets.flatten()
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        return np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.7)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Sharpe ratios
        axes[1].plot(self.train_sharpes, label='Train Sharpe', alpha=0.7)
        axes[1].plot(self.val_sharpes, label='Val Sharpe', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Training and Validation Sharpe Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    from data_collection import HyperliquidDataCollector
    from feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("ML MODELS MODULE TEST")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    collector = HyperliquidDataCollector(use_synthetic=True)
    df = collector.get_ohlcv('BTC', interval='1h', lookback_hours=720)
    df = collector.calculate_variance_metrics(df)
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    train_df, val_df, test_df = engineer.prepare_ml_dataset(df_features)
    
    # Prepare data loaders
    print("\n3. Preparing data loaders...")
    train_dataset = TradingDataset(
        train_df[engineer.feature_columns].values,
        train_df['target'].values
    )
    val_dataset = TradingDataset(
        val_df[engineer.feature_columns].values,
        val_df['target'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\n4. Creating model...")
    model = TradingNN(
        input_size=len(engineer.feature_columns),
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n5. Training model...")
    trainer = ModelTrainer(model, SharpeLoss(), learning_rate=0.001)
    history = trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=15)
    
    # Plot results
    trainer.plot_training_history()
    
    # Generate predictions
    print("\n6. Generating predictions...")
    test_dataset = TradingDataset(
        test_df[engineer.feature_columns].values,
        test_df['target'].values
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = trainer.predict(test_loader)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("ML MODELS TEST COMPLETE")
    print("=" * 60)
