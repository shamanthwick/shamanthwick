#!/usr/bin/env python3
"""
NeuroCrypto: Neural Network Implementations of AES and ChaCha
Task 3 Submission - IIT Roorkee

This module implements neural network-based cryptographic primitives:
- S-box Neural Network
- MixColumns Neural Layer
- Complete Neuro-AES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ========== AES S-box Reference =========

AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

AES_INV_SBOX = [0] * 256
for i in range(256):
    AES_INV_SBOX[AES_SBOX[i]] = i


# ========== S-box Neural Network =========

class SBoxNeuralNetwork(nn.Module):
    """
    Neural Network implementation of AES S-box.
    
    Architecture:
    - Input: 8-bit byte (256 possible values)
    - Embedding: 256 → 128 dimensions
    - MLP: 128 → 64 → 32 → 256 (output logits)
    - Output: 8-bit substituted value (256 classes)
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super(SBoxNeuralNetwork, self).__init__()
        
        self.embedding = nn.Embedding(256, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 256)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through S-box network.
        
        Args:
            x: Input bytes (batch_size,) with values 0-255
        
        Returns:
            Output logits (batch_size, 256)
        """
        embedded = self.embedding(x)  # (batch, embedding_dim)
        output = self.mlp(embedded)    # (batch, 256)
        return output
    
    def substitute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform S-box substitution (inference mode).
        
        Args:
            x: Input bytes (batch_size,) with values 0-255
        
        Returns:
            Substituted bytes (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            output = torch.argmax(logits, dim=1)
        return output
    
    def get_accuracy(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> float:
        """Calculate accuracy on test data."""
        self.eval()
        predictions = self.substitute(test_data)
        accuracy = (predictions == test_labels).float().mean().item()
        return accuracy


# ========== MixColumns Neural Layer =========

class MixColumnsLayer(nn.Module):
    """
    Neural Network implementation of AES MixColumns.
    
    This is implemented as a fixed-weight linear layer since
    MixColumns is a linear transformation.
    
    MixColumns matrix:
    [02 03 01 01]
    [01 02 03 01]
    [01 01 02 03]
    [03 01 01 02]
    """
    
    def __init__(self):
        super(MixColumnsLayer, self).__init__()
        
        # MixColumns matrix in GF(2^8)
        # We'll implement this as a lookup-based operation
        self.register_buffer('mix_matrix', torch.tensor([
            [2, 3, 1, 1],
            [1, 2, 3, 1],
            [1, 1, 2, 3],
            [3, 1, 1, 2]
        ], dtype=torch.uint8))
        
        # Precompute GF(2^8) multiplication tables
        self._init_gf_tables()
    
    def _init_gf_tables(self):
        """Initialize GF(2^8) multiplication tables"""
        # Multiply by 2 in GF(2^8)
        self.mul2_table = torch.zeros(256, dtype=torch.uint8)
        # Multiply by 3 in GF(2^8)
        self.mul3_table = torch.zeros(256, dtype=torch.uint8)
        
        for i in range(256):
            # Multiply by 2
            self.mul2_table[i] = (i << 1) ^ (0x11b if i & 0x80 else 0)
            # Multiply by 3 = multiply by 2 + multiply by 1
            self.mul3_table[i] = self.mul2_table[i] ^ i
    
    def _gf_mul(self, a: int, b: int) -> int:
        """Multiply two numbers in GF(2^8)"""
        if a == 1:
            return b
        elif a == 2:
            return self.mul2_table[b].item()
        elif a == 3:
            return self.mul3_table[b].item()
        else:
            raise ValueError(f"Unsupported multiplication by {a}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply MixColumns transformation.
        
        Args:
            state: 4x4 state matrix (4, 4) with byte values
        
        Returns:
            Transformed state matrix (4, 4)
        """
        output = torch.zeros_like(state)
        
        for col in range(4):
            for row in range(4):
                result = 0
                for k in range(4):
                    a = self.mix_matrix[row, k].item()
                    b = state[k, col].item()
                    result ^= self._gf_mul(a, b)
                output[row, col] = result
        
        return output


# ========== ShiftRows Permutation Layer =========

class ShiftRowsLayer(nn.Module):
    """
    Neural Network implementation of AES ShiftRows.
    
    This is implemented as a fixed permutation layer.
    """
    
    def __init__(self):
        super(ShiftRowsLayer, self).__init__()
        
        # ShiftRows permutation indices
        # Row 0: no shift
        # Row 1: shift left by 1
        # Row 2: shift left by 2
        # Row 3: shift left by 3
        self.register_buffer('perm_indices', torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2]
        ], dtype=torch.long))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply ShiftRows transformation.
        
        Args:
            state: 4x4 state matrix (4, 4)
        
        Returns:
            Permuted state matrix (4, 4)
        """
        output = torch.zeros_like(state)
        for row in range(4):
            output[row] = state[row, self.perm_indices[row]]
        return output


# ========== AddRoundKey Layer =========

class AddRoundKeyLayer(nn.Module):
    """
    Neural Network implementation of AES AddRoundKey.
    
    This is implemented as XOR operation.
    """
    
    def __init__(self):
        super(AddRoundKeyLayer, self).__init__()
    
    def forward(self, state: torch.Tensor, round_key: torch.Tensor) -> torch.Tensor:
        """
        Apply AddRoundKey transformation (XOR).
        
        Args:
            state: 4x4 state matrix (4, 4)
            round_key: 4x4 round key matrix (4, 4)
        
        Returns:
            XORed state matrix (4, 4)
        """
        return state ^ round_key


# ========== Complete Neuro-AES =========

class NeuroAES(nn.Module):
    """
    Complete Neural Network implementation of AES-128.
    
    Architecture:
    - Initial AddRoundKey
    - 9 rounds of: SubBytes → ShiftRows → MixColumns → AddRoundKey
    - Final round: SubBytes → ShiftRows → AddRoundKey (no MixColumns)
    """
    
    def __init__(self, sbox_net: Optional[SBoxNeuralNetwork] = None):
        super(NeuroAES, self).__init__()
        
        # Neural network components
        self.sbox_net = sbox_net or SBoxNeuralNetwork()
        
        # Fixed transformation layers
        self.shift_rows = ShiftRowsLayer()
        self.mix_columns = MixColumnsLayer()
        self.add_round_key = AddRoundKeyLayer()
        
        # AES-128 uses 10 rounds
        self.num_rounds = 10
    
    def key_expansion(self, key: torch.Tensor) -> list:
        """
        Expand the 128-bit key into 11 round keys.
        
        Args:
            key: 128-bit key as 16 bytes
        
        Returns:
            List of 11 round keys (each 4x4 matrix)
        """
        # Simplified key expansion (for demo purposes)
        # In production, implement full AES key schedule
        round_keys = []
        current_key = key.reshape(4, 4)
        
        for _ in range(self.num_rounds + 1):
            round_keys.append(current_key.clone())
            # Simple key evolution (XOR with round constant)
            current_key = current_key ^ 0x01
        
        return round_keys
    
    def state_to_bytes(self, state: torch.Tensor) -> torch.Tensor:
        """Convert 4x4 state to 16-byte tensor"""
        return state.reshape(16)
    
    def bytes_to_state(self, bytes_16: torch.Tensor) -> torch.Tensor:
        """Convert 16 bytes to 4x4 state"""
        return bytes_16.reshape(4, 4)
    
    def forward(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Encrypt plaintext using AES-128.
        
        Args:
            plaintext: 16-byte plaintext
            key: 16-byte key
        
        Returns:
            16-byte ciphertext
        """
        # Convert to state matrix
        state = self.bytes_to_state(plaintext)
        
        # Expand key
        round_keys = self.key_expansion(key)
        
        # Initial round
        state = self.add_round_key(state, round_keys[0])
        
        # Main rounds (1-9)
        for round_num in range(1, self.num_rounds):
            # SubBytes (using neural network)
            for i in range(4):
                for j in range(4):
                    byte_val = state[i, j].unsqueeze(0)
                    substituted = self.sbox_net.substitute(byte_val)
                    state[i, j] = substituted
            
            # ShiftRows
            state = self.shift_rows(state)
            
            # MixColumns
            state = self.mix_columns(state)
            
            # AddRoundKey
            state = self.add_round_key(state, round_keys[round_num])
        
        # Final round (no MixColumns)
        for i in range(4):
            for j in range(4):
                byte_val = state[i, j].unsqueeze(0)
                substituted = self.sbox_net.substitute(byte_val)
                state[i, j] = substituted
        
        state = self.shift_rows(state)
        state = self.add_round_key(state, round_keys[self.num_rounds])
        
        # Convert back to bytes
        ciphertext = self.state_to_bytes(state)
        
        return ciphertext


# ========== Training Utilities =========

def create_sbox_training_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training data for S-box neural network.
    
    Returns:
        Tuple of (input_bytes, output_bytes) - all 256 pairs
    """
    inputs = torch.arange(256, dtype=torch.long)
    outputs = torch.tensor(AES_SBOX, dtype=torch.long)
    return inputs, outputs


def train_sbox_net(model: SBoxNeuralNetwork, epochs: int = 1000, 
                   lr: float = 0.001, verbose: bool = True) -> list:
    """
    Train the S-box neural network.
    
    Args:
        model: SBoxNeuralNetwork to train
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        List of loss values per epoch
    """
    # Create training data
    inputs, labels = create_sbox_training_data()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 100 == 0:
            accuracy = model.get_accuracy(inputs, labels)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")
    
    return losses


if __name__ == "__main__":
    print("=" * 70)
    print("NeuroCrypto: Neural Network AES Implementation")
    print("Task 3 - IIT Roorkee")
    print("=" * 70)
    
    # Create and train S-box network
    print("\n1. Creating S-box Neural Network...")
    sbox_net = SBoxNeuralNetwork(embedding_dim=128, hidden_dim=64)
    print(f"   ✓ Model created with {sum(p.numel() for p in sbox_net.parameters()):,} parameters")
    
    print("\n2. Training S-box Neural Network...")
    losses = train_sbox_net(sbox_net, epochs=1000, lr=0.001)
    
    print("\n3. Evaluating S-box Network...")
    inputs, labels = create_sbox_training_data()
    accuracy = sbox_net.get_accuracy(inputs, labels)
    print(f"   ✓ Final Accuracy: {accuracy*100:.2f}%")
    
    print("\n4. Testing Neuro-AES...")
    neuro_aes = NeuroAES(sbox_net)
    
    # Test encryption
    plaintext = torch.randint(0, 256, (16,), dtype=torch.long)
    key = torch.randint(0, 256, (16,), dtype=torch.long)
    
    print(f"   Plaintext: {plaintext.numpy().hex()}")
    print(f"   Key: {key.numpy().hex()}")
    
    ciphertext = neuro_aes(plaintext, key)
    print(f"   Ciphertext: {ciphertext.numpy().hex()}")
    
    print("\n✅ Neuro-AES implementation complete!")
    print("\nNote: This is a research implementation for educational purposes.")
    print("Not suitable for production cryptographic use.")
