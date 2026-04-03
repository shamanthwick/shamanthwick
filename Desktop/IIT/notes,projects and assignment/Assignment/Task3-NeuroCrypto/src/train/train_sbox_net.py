#!/usr/bin/env python3
"""
Train S-box Neural Network
Task 3 Submission - IIT Roorkee

This script trains a neural network to learn the AES S-box function.

Usage:
    python train_sbox_net.py                  # Train with default settings
    python train_sbox_net.py --epochs 2000    # Custom epoch count
    python train_sbox_net.py --lr 0.0005      # Custom learning rate
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

from models.neuro_aes import SBoxNeuralNetwork, train_sbox_net, create_sbox_training_data


def main():
    parser = argparse.ArgumentParser(description='Train S-box Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Training S-box Neural Network")
    print("Task 3 - IIT Roorkee")
    print("="*70)
    
    # Create output directories
    os.makedirs('../../models', exist_ok=True)
    os.makedirs('../../results', exist_ok=True)
    
    # Create model
    print("\n1. Creating S-box Neural Network...")
    model = SBoxNeuralNetwork(embedding_dim=128, hidden_dim=64)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {num_params:,} parameters")
    
    # Train
    print(f"\n2. Training for {args.epochs} epochs...")
    print("   This may take a few minutes...")
    losses = train_sbox_net(model, epochs=args.epochs, lr=args.lr, verbose=True)
    
    # Evaluate
    print("\n3. Evaluating Model...")
    inputs, labels = create_sbox_training_data()
    accuracy = model.get_accuracy(inputs, labels)
    print(f"   ✓ Final Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    print("\n4. Saving Model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': 128,
        'hidden_dim': 64,
        'accuracy': accuracy,
        'num_parameters': num_params,
        'epochs_trained': args.epochs
    }, '../../models/sbox_net.pth')
    print(f"   ✓ Model saved to models/sbox_net.pth")
    
    # Save metrics
    metrics = {
        'model': 'SBoxNeuralNetwork',
        'timestamp': datetime.now().isoformat(),
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'final_loss': losses[-1],
        'final_accuracy': accuracy,
        'num_parameters': num_params,
        'losses': losses
    }
    
    with open('../../results/sbox_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to results/sbox_training_metrics.json")
    
    # Plot training curve
    print("\n5. Plotting Training Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('S-box Neural Network Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('../../results/sbox_training_curve.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to results/sbox_training_curve.png")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"""
Summary:
  - Model: SBoxNeuralNetwork
  - Parameters: {num_params:,}
  - Epochs: {args.epochs}
  - Final Loss: {losses[-1]:.6f}
  - Final Accuracy: {accuracy*100:.2f}%

Output Files:
  - models/sbox_net.pth (trained model)
  - results/sbox_training_metrics.json (metrics)
  - results/sbox_training_curve.png (training curve)
""")
    
    print("⚠️  Note: This is a research implementation for educational purposes.")
    print("   NOT suitable for production cryptographic use.")
    print("="*70)


if __name__ == "__main__":
    main()
