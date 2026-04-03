#!/usr/bin/env python3
"""
Main Training Script for NeuroCrypto
Task 3 Submission - IIT Roorkee

This script trains all neural network models:
1. S-box Neural Network (for AES)
2. Neural Quarter Round (for ChaCha20)

Usage:
    python train_all.py                  # Train all models with default settings
    python train_all.py --sbox-only      # Train only S-box network
    python train_all.py --chacha-only    # Train only ChaCha quarter round
    python train_all.py --epochs 2000    # Custom epoch count
"""

import torch
import json
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neuro_aes import SBoxNeuralNetwork, train_sbox_net, create_sbox_training_data
from models.neuro_chacha import NeuralQuarterRound, train_neural_chacha


def train_sbox_model(epochs=1000, lr=0.001, save_dir='../../models'):
    """Train the S-box neural network"""
    print("\n" + "="*70)
    print("TRAINING S-BOX NEURAL NETWORK")
    print("="*70)
    
    # Create model
    print("\n1. Creating S-box Neural Network...")
    model = SBoxNeuralNetwork(embedding_dim=128, hidden_dim=64)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {num_params:,} parameters")
    
    # Train
    print(f"\n2. Training for {epochs} epochs...")
    losses = train_sbox_net(model, epochs=epochs, lr=lr, verbose=True)
    
    # Evaluate
    print("\n3. Evaluating Model...")
    inputs, labels = create_sbox_training_data()
    accuracy = model.get_accuracy(inputs, labels)
    print(f"   ✓ Final Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'sbox_net.pth')
    print(f"\n4. Saving Model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': 128,
        'hidden_dim': 64,
        'accuracy': accuracy,
        'num_parameters': num_params,
        'epochs_trained': epochs
    }, model_path)
    print(f"   ✓ Model saved")
    
    # Save metrics
    metrics = {
        'model': 'SBoxNeuralNetwork',
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'learning_rate': lr,
        'final_loss': losses[-1],
        'final_accuracy': accuracy,
        'num_parameters': num_params,
        'losses': losses
    }
    
    metrics_path = os.path.join(save_dir.replace('models', 'results'), 'sbox_training_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to {metrics_path}")
    
    # Plot training curve
    print("\n5. Plotting Training Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('S-box Neural Network Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.join(save_dir.replace('models', 'results'), 'sbox_training_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Plot saved to {plot_path}")
    
    print("\n✅ S-box Training Complete!")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Parameters: {num_params:,}")
    
    return model, metrics


def train_chacha_model(epochs=100, lr=0.001, save_dir='../../models'):
    """Train the ChaCha neural quarter round"""
    print("\n" + "="*70)
    print("TRAINING NEURAL CHACHA QUARTER ROUND")
    print("="*70)
    
    # Create model
    print("\n1. Creating Neural Quarter Round...")
    model = NeuralQuarterRound(hidden_dim=256)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {num_params:,} parameters")
    
    # Train
    print(f"\n2. Training for {epochs} epochs...")
    losses = train_neural_chacha(model, epochs=epochs, lr=lr, verbose=True)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'chacha_quarter_round.pth')
    print(f"\n3. Saving Model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': 256,
        'num_parameters': num_params,
        'epochs_trained': epochs,
        'final_loss': losses[-1]
    }, model_path)
    print(f"   ✓ Model saved")
    
    # Save metrics
    metrics = {
        'model': 'NeuralQuarterRound',
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'learning_rate': lr,
        'final_loss': losses[-1],
        'num_parameters': num_params,
        'losses': losses
    }
    
    metrics_path = os.path.join(save_dir.replace('models', 'results'), 'chacha_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to {metrics_path}")
    
    # Plot training curve
    print("\n4. Plotting Training Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Neural ChaCha Quarter Round Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.join(save_dir.replace('models', 'results'), 'chacha_training_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Plot saved to {plot_path}")
    
    print("\n✅ ChaCha Quarter Round Training Complete!")
    print(f"   Parameters: {num_params:,}")
    print(f"   Final Loss: {losses[-1]:.6f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train NeuroCrypto Models')
    parser.add_argument('--sbox-only', action='store_true', help='Train only S-box network')
    parser.add_argument('--chacha-only', action='store_true', help='Train only ChaCha quarter round')
    parser.add_argument('--sbox-epochs', type=int, default=1000, help='Epochs for S-box training (default: 1000)')
    parser.add_argument('--chacha-epochs', type=int, default=100, help='Epochs for ChaCha training (default: 100)')
    parser.add_argument('--sbox-lr', type=float, default=0.001, help='Learning rate for S-box (default: 0.001)')
    parser.add_argument('--chacha-lr', type=float, default=0.001, help='Learning rate for ChaCha (default: 0.001)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NeuroCrypto: Training All Models")
    print("Task 3 - IIT Roorkee")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  S-box epochs: {args.sbox_epochs}")
    print(f"  S-box learning rate: {args.sbox_lr}")
    print(f"  ChaCha epochs: {args.chacha_epochs}")
    print(f"  ChaCha learning rate: {args.chacha_lr}")
    
    # Determine what to train
    train_sbox = not args.chacha_only
    train_chacha = not args.sbox_only
    
    results = {}
    
    # Train S-box
    if train_sbox:
        sbox_model, sbox_metrics = train_sbox_model(
            epochs=args.sbox_epochs,
            lr=args.sbox_lr
        )
        results['sbox'] = sbox_metrics
    
    # Train ChaCha
    if train_chacha:
        chacha_model, chacha_metrics = train_chacha_model(
            epochs=args.chacha_epochs,
            lr=args.chacha_lr
        )
        results['chacha'] = chacha_metrics
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    if train_sbox:
        print(f"\n✅ S-box Network:")
        print(f"   Accuracy: {results['sbox']['final_accuracy']*100:.2f}%")
        print(f"   Parameters: {results['sbox']['num_parameters']:,}")
        print(f"   Final Loss: {results['sbox']['final_loss']:.6f}")
    
    if train_chacha:
        print(f"\n✅ ChaCha Quarter Round:")
        print(f"   Parameters: {results['chacha']['num_parameters']:,}")
        print(f"   Final Loss: {results['chacha']['final_loss']:.6f}")
    
    print("\n" + "="*70)
    print("Output Files:")
    print("="*70)
    
    if train_sbox:
        print("  models/sbox_net.pth")
        print("  results/sbox_training_metrics.json")
        print("  results/sbox_training_curve.png")
    
    if train_chacha:
        print("  models/chacha_quarter_round.pth")
        print("  results/chacha_training_metrics.json")
        print("  results/chacha_training_curve.png")
    
    print("\n" + "="*70)
    print("⚠️  Note: This is a research implementation for educational purposes.")
    print("   NOT suitable for production cryptographic use.")
    print("="*70)


if __name__ == "__main__":
    main()
