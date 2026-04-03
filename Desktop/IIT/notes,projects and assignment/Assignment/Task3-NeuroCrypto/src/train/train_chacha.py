#!/usr/bin/env python3
"""
Train ChaCha Neural Quarter Round
Task 3 Submission - IIT Roorkee

This script trains a neural network to approximate ChaCha20's ARX operations.

Usage:
    python train_chacha.py                  # Train with default settings
    python train_chacha.py --epochs 200     # Custom epoch count
    python train_chacha.py --lr 0.0005      # Custom learning rate
"""

import torch
import json
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neuro_chacha import NeuralQuarterRound, train_neural_chacha


def main():
    parser = argparse.ArgumentParser(description='Train ChaCha Neural Quarter Round')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Training ChaCha Neural Quarter Round")
    print("Task 3 - IIT Roorkee")
    print("="*70)
    
    # Create output directories
    os.makedirs('../../models', exist_ok=True)
    os.makedirs('../../results', exist_ok=True)
    
    # Create model
    print("\n1. Creating Neural Quarter Round...")
    model = NeuralQuarterRound(hidden_dim=256)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {num_params:,} parameters")
    
    # Train
    print(f"\n2. Training for {args.epochs} epochs...")
    print("   This may take several minutes...")
    losses = train_neural_chacha(model, epochs=args.epochs, lr=args.lr, verbose=True)
    
    # Save model
    print("\n3. Saving Model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': 256,
        'num_parameters': num_params,
        'epochs_trained': args.epochs,
        'final_loss': losses[-1]
    }, '../../models/chacha_quarter_round.pth')
    print(f"   ✓ Model saved to models/chacha_quarter_round.pth")
    
    # Save metrics
    metrics = {
        'model': 'NeuralQuarterRound',
        'timestamp': datetime.now().isoformat(),
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'final_loss': losses[-1],
        'num_parameters': num_params,
        'losses': losses
    }
    
    with open('../../results/chacha_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to results/chacha_training_metrics.json")
    
    # Plot training curve
    print("\n4. Plotting Training Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural ChaCha Quarter Round Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('../../results/chacha_training_curve.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to results/chacha_training_curve.png")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"""
Summary:
  - Model: NeuralQuarterRound
  - Parameters: {num_params:,}
  - Epochs: {args.epochs}
  - Final Loss: {losses[-1]:.6f}

Output Files:
  - models/chacha_quarter_round.pth (trained model)
  - results/chacha_training_metrics.json (metrics)
  - results/chacha_training_curve.png (training curve)
""")
    
    print("⚠️  Note: This is a research implementation for educational purposes.")
    print("   NOT suitable for production cryptographic use.")
    print("="*70)


if __name__ == "__main__":
    main()
