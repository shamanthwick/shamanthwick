#!/usr/bin/env python3
"""
Evaluate Trained NeuroCrypto Models
Task 3 Submission - IIT Roorkee

This script loads trained models and evaluates their performance.

Usage:
    python evaluate_models.py
"""

import torch
import json
import os
import sys
import time
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neuro_aes import SBoxNeuralNetwork, NeuroAES, AES_SBOX
from models.neuro_chacha import NeuralQuarterRound


def evaluate_sbox_model(model_path='../../models/sbox_net.pth'):
    """Evaluate the trained S-box network"""
    print("\n" + "="*70)
    print("EVALUATING S-BOX NEURAL NETWORK")
    print("="*70)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"   ❌ Model file not found: {model_path}")
        print("   Please train the model first using: python train_sbox_net.py")
        return None
    
    checkpoint = torch.load(model_path, weights_only=False)
    model = SBoxNeuralNetwork(
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Parameters: {checkpoint['num_parameters']:,}")
    print(f"   ✓ Training accuracy: {checkpoint['accuracy']*100:.2f}%")
    
    # Test on all 256 inputs
    print("\n2. Testing on all 256 input-output pairs...")
    inputs = torch.arange(256, dtype=torch.long)
    labels = torch.tensor(AES_SBOX, dtype=torch.long)
    
    with torch.no_grad():
        predictions = model.substitute(inputs)
    
    accuracy = (predictions == labels).float().mean().item()
    print(f"   ✓ Test Accuracy: {accuracy*100:.2f}%")
    
    # Check which inputs failed (if any)
    failures = (predictions != labels).nonzero(as_tuple=True)[0]
    if len(failures) > 0:
        print(f"\n   ⚠️  {len(failures)} failures detected:")
        for idx in failures[:10]:  # Show first 10
            print(f"      Input {idx.item():3d}: Expected {labels[idx]:3d}, Got {predictions[idx]:3d}")
        if len(failures) > 10:
            print(f"      ... and {len(failures) - 10} more")
    else:
        print(f"   ✅ Perfect! All 256 pairs correct!")
    
    # Inference speed test
    print("\n3. Testing inference speed...")
    test_input = torch.randint(0, 256, (1000,), dtype=torch.long)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model.substitute(test_input)
    elapsed = time.time() - start
    total_bytes = 1000 * 100
    speed = total_bytes / elapsed
    print(f"   ✓ Processed {total_bytes:,} bytes in {elapsed:.3f}s")
    print(f"   ✓ Speed: {speed:,.0f} bytes/second")
    print(f"   ✓ Time per byte: {elapsed/(total_bytes)*1e6:.2f} μs")
    
    return {
        'accuracy': accuracy,
        'num_failures': len(failures),
        'speed_bytes_per_sec': speed,
        'time_per_byte_us': elapsed/(total_bytes)*1e6
    }


def evaluate_chacha_model(model_path='../../models/chacha_quarter_round.pth'):
    """Evaluate the trained ChaCha quarter round"""
    print("\n" + "="*70)
    print("EVALUATING NEURAL CHACHA QUARTER ROUND")
    print("="*70)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"   ❌ Model file not found: {model_path}")
        print("   Please train the model first using: python train_chacha.py")
        return None
    
    checkpoint = torch.load(model_path, weights_only=False)
    model = NeuralQuarterRound(hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Parameters: {checkpoint['num_parameters']:,}")
    print(f"   ✓ Training loss: {checkpoint['final_loss']:.6f}")
    
    # Test on random inputs
    print("\n2. Testing on random inputs...")
    num_tests = 100
    test_inputs = torch.randint(0, 2**32, (num_tests, 4), dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(test_inputs)
    
    # Check output shape and range
    print(f"   ✓ Input shape: {test_inputs.shape}")
    print(f"   ✓ Output shape: {outputs.shape}")
    print(f"   ✓ Output range: [{outputs.min().item():.0f}, {outputs.max().item():.0f}]")
    
    # Inference speed test
    print("\n3. Testing inference speed...")
    test_input = torch.randint(0, 2**32, (100, 4), dtype=torch.long)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    elapsed = time.time() - start
    total_ops = 100 * 100
    speed = total_ops / elapsed
    print(f"   ✓ Processed {total_ops:,} quarter rounds in {elapsed:.3f}s")
    print(f"   ✓ Speed: {speed:,.0f} quarter rounds/second")
    print(f"   ✓ Time per quarter round: {elapsed/(total_ops)*1e3:.2f} ms")
    
    return {
        'speed_ops_per_sec': speed,
        'time_per_op_ms': elapsed/(total_ops)*1e3
    }


def evaluate_neuro_aes(sbox_model_path='../../models/sbox_net.pth'):
    """Evaluate complete Neuro-AES"""
    print("\n" + "="*70)
    print("EVALUATING COMPLETE NEURO-AES")
    print("="*70)
    
    # Load S-box model
    if not os.path.exists(sbox_model_path):
        print(f"   ❌ S-box model not found: {sbox_model_path}")
        return None
    
    checkpoint = torch.load(sbox_model_path, weights_only=False)
    sbox = SBoxNeuralNetwork(
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    sbox.load_state_dict(checkpoint['model_state_dict'])
    
    # Create Neuro-AES
    print("\n1. Creating Neuro-AES...")
    aes = NeuroAES(sbox)
    aes.eval()
    print(f"   ✓ Neuro-AES created")
    
    # Test encryption
    print("\n2. Testing encryption...")
    plaintext = torch.randint(0, 256, (16,), dtype=torch.long)
    key = torch.randint(0, 256, (16,), dtype=torch.long)
    
    print(f"   Plaintext: {plaintext.numpy().tolist()}")
    print(f"   Key:       {key.numpy().tolist()}")
    
    with torch.no_grad():
        start = time.time()
        ciphertext = aes(plaintext, key)
        enc_time = time.time() - start
    
    print(f"   Ciphertext: {ciphertext.numpy().tolist()}")
    print(f"   ✓ Encryption time: {enc_time*1000:.2f} ms")
    
    # Test determinism
    print("\n3. Testing determinism (same input → same output)...")
    with torch.no_grad():
        ciphertext2 = aes(plaintext, key)
    
    if torch.equal(ciphertext, ciphertext2):
        print(f"   ✅ Deterministic! Same plaintext + key = same ciphertext")
    else:
        print(f"   ❌ NOT deterministic! This is a problem!")
    
    # Test avalanche effect (small change in input → large change in output)
    print("\n4. Testing avalanche effect...")
    plaintext_modified = plaintext.clone()
    plaintext_modified[0] ^= 0x01  # Flip 1 bit
    
    with torch.no_grad():
        ciphertext_modified = aes(plaintext_modified, key)
    
    diff_bits = (ciphertext != ciphertext_modified).sum().item()
    print(f"   Changed 1 bit in plaintext")
    print(f"   Changed {diff_bits} bits in ciphertext (out of 128)")
    print(f"   Avalanche ratio: {diff_bits/128*100:.1f}%")
    if diff_bits > 40:  # Good avalanche
        print(f"   ✅ Good avalanche effect!")
    else:
        print(f"   ⚠️  Weak avalanche effect")
    
    return {
        'encryption_time_ms': enc_time * 1000,
        'deterministic': torch.equal(ciphertext, ciphertext2),
        'avalanche_bits': diff_bits
    }


def main():
    print("="*70)
    print("NeuroCrypto: Model Evaluation")
    print("Task 3 - IIT Roorkee")
    print("="*70)
    
    results = {}
    
    # Evaluate S-box
    sbox_results = evaluate_sbox_model()
    if sbox_results:
        results['sbox'] = sbox_results
    
    # Evaluate ChaCha
    chacha_results = evaluate_chacha_model()
    if chacha_results:
        results['chacha'] = chacha_results
    
    # Evaluate Neuro-AES
    aes_results = evaluate_neuro_aes()
    if aes_results:
        results['aes'] = aes_results
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    if 'sbox' in results:
        print(f"\n✅ S-box Network:")
        print(f"   Accuracy: {results['sbox']['accuracy']*100:.2f}%")
        print(f"   Speed: {results['sbox']['speed_bytes_per_sec']:,.0f} bytes/sec")
    
    if 'chacha' in results:
        print(f"\n✅ ChaCha Quarter Round:")
        print(f"   Speed: {results['chacha']['speed_ops_per_sec']:,.0f} ops/sec")
    
    if 'aes' in results:
        print(f"\n✅ Neuro-AES:")
        print(f"   Encryption time: {results['aes']['encryption_time_ms']:.2f} ms")
        print(f"   Deterministic: {results['aes']['deterministic']}")
        print(f"   Avalanche: {results['aes']['avalanche_bits']} bits changed")
    
    # Save results
    os.makedirs('../../results', exist_ok=True)
    with open('../../results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to results/evaluation_results.json")
    print("\n" + "="*70)
    print("⚠️  Note: This is a research implementation for educational purposes.")
    print("   NOT suitable for production cryptographic use.")
    print("="*70)


if __name__ == "__main__":
    main()
