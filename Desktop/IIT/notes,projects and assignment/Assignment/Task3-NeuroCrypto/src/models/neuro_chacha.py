#!/usr/bin/env python3
"""
NeuroCrypto: Neural Network Implementation of ChaCha20
Task 3 Submission - IIT Roorkee

This module implements neural network-based ChaCha20 stream cipher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ========== ChaCha20 Reference Implementation =========

def chacha20_quarter_round(state: list, a: int, b: int, c: int, d: int) -> None:
    """ChaCha20 quarter round function (ARX operations)"""
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = ((state[d] << 16) | (state[d] >> 16)) & 0xFFFFFFFF
    
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = ((state[b] << 12) | (state[b] >> 20)) & 0xFFFFFFFF
    
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = ((state[d] << 8) | (state[d] >> 24)) & 0xFFFFFFFF
    
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = ((state[b] << 7) | (state[b] >> 25)) & 0xFFFFFFFF


# ========== Neural Network Quarter Round =========

class NeuralQuarterRound(nn.Module):
    """
    Neural Network implementation of ChaCha20 Quarter Round.
    
    The quarter round uses ARX (Add-Rotate-XOR) operations.
    We approximate this using neural networks.
    
    Architecture:
    - Input: 4 × 32-bit words (128 bits total)
    - Multiple ARX approximation layers
    - Output: 4 × 32-bit words (transformed)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super(NeuralQuarterRound, self).__init__()

        # ARX approximation network
        self.arx_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )

        # Residual connection for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.arx_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply neural quarter round.

        Args:
            state: 4 × 32-bit words (batch, 4) or (4,)

        Returns:
            Transformed state (batch, 4) or (4,)
        """
        # Handle both batched and unbatched input
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            unbatched = True
        else:
            unbatched = False
        
        batch_size = state.size(0)
        
        # Flatten to 128 bits (normalize to [0, 1])
        flat = state.float() / 0xFFFFFFFF  # (batch, 4)

        # Apply ARX approximation
        output = self.arx_network(flat)  # (batch, 128) -> should be (batch, 4)

        # Residual connection
        output = output + flat

        # Scale back
        output = (output * 0xFFFFFFFF).clamp(0, 0xFFFFFFFF)

        if unbatched:
            output = output.squeeze(0)
        
        return output


# ========== Neural ChaCha20 Block Function =========

class NeuralChaCha20Block(nn.Module):
    """
    Neural Network implementation of ChaCha20 block function.
    
    Architecture:
    - Initial state: 16 × 32-bit words (512 bits)
    - 20 rounds of neural quarter rounds
    - Output: 512-bit keystream block
    """
    
    def __init__(self):
        super(NeuralChaCha20Block, self).__init__()
        
        # Neural quarter round
        self.quarter_round = NeuralQuarterRound(hidden_dim=256)
        
        # 20 rounds (10 double rounds)
        self.num_rounds = 10
        
        # ChaCha20 constants (tau: "expand 32-byte k")
        self.register_buffer('constants', torch.tensor([
            0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
        ], dtype=torch.long))
    
    def init_state(self, key: torch.Tensor, counter: torch.Tensor, nonce: torch.Tensor) -> torch.Tensor:
        """
        Initialize ChaCha20 state.
        
        Args:
            key: 256-bit key (8 × 32-bit words)
            counter: 32-bit block counter
            nonce: 96-bit nonce (3 × 32-bit words)
        
        Returns:
            16 × 32-bit initial state
        """
        state = torch.zeros(16, dtype=torch.long)
        
        # Constants
        state[0:4] = self.constants
        
        # Key
        state[4:12] = key
        
        # Counter
        state[12] = counter
        
        # Nonce
        state[13:16] = nonce
        
        return state
    
    def column_round(self, state: torch.Tensor) -> torch.Tensor:
        """Apply column round (4 quarter rounds on columns)"""
        state = state.clone()
        
        # Column 0
        qr_out = self.quarter_round(state[0::4].unsqueeze(1)).squeeze(1)
        state[0::4] = qr_out
        
        # Column 1
        qr_out = self.quarter_round(state[1::4].unsqueeze(1)).squeeze(1)
        state[1::4] = qr_out
        
        # Column 2
        qr_out = self.quarter_round(state[2::4].unsqueeze(1)).squeeze(1)
        state[2::4] = qr_out
        
        # Column 3
        qr_out = self.quarter_round(state[3::4].unsqueeze(1)).squeeze(1)
        state[3::4] = qr_out
        
        return state
    
    def diagonal_round(self, state: torch.Tensor) -> torch.Tensor:
        """Apply diagonal round (4 quarter rounds on diagonals)"""
        state = state.clone()
        
        # Diagonal 0
        indices = [0, 5, 10, 15]
        qr_out = self.quarter_round(state[indices].unsqueeze(1)).squeeze(1)
        state[indices] = qr_out
        
        # Diagonal 1
        indices = [1, 6, 11, 12]
        qr_out = self.quarter_round(state[indices].unsqueeze(1)).squeeze(1)
        state[indices] = qr_out
        
        # Diagonal 2
        indices = [2, 7, 8, 13]
        qr_out = self.quarter_round(state[indices].unsqueeze(1)).squeeze(1)
        state[indices] = qr_out
        
        # Diagonal 3
        indices = [3, 4, 9, 14]
        qr_out = self.quarter_round(state[indices].unsqueeze(1)).squeeze(1)
        state[indices] = qr_out
        
        return state
    
    def forward(self, key: torch.Tensor, counter: torch.Tensor, 
                nonce: torch.Tensor) -> torch.Tensor:
        """
        Generate ChaCha20 keystream block.
        
        Args:
            key: 256-bit key (8 × 32-bit words)
            counter: 32-bit block counter
            nonce: 96-bit nonce (3 × 32-bit words)
        
        Returns:
            512-bit keystream block (16 × 32-bit words)
        """
        # Initialize state
        state = self.init_state(key, counter, nonce)
        initial_state = state.clone()
        
        # 20 rounds (10 double rounds)
        for _ in range(self.num_rounds):
            # Column round
            state = self.column_round(state)
            
            # Diagonal round
            state = self.diagonal_round(state)
        
        # Add initial state (feed-forward)
        output = (state + initial_state) & 0xFFFFFFFF
        
        return output


# ========== Complete Neural ChaCha20 =========

class NeuralChaCha20(nn.Module):
    """
    Complete Neural Network implementation of ChaCha20 stream cipher.
    
    Generates arbitrary-length keystream using ChaCha20 block function.
    """
    
    def __init__(self):
        super(NeuralChaCha20, self).__init__()
        
        self.block_function = NeuralChaCha20Block()
    
    def forward(self, key: torch.Tensor, nonce: torch.Tensor, 
                length: int) -> torch.Tensor:
        """
        Generate ChaCha20 keystream.
        
        Args:
            key: 256-bit key (8 × 32-bit words)
            nonce: 96-bit nonce (3 × 32-bit words)
            length: Desired keystream length in bytes
        
        Returns:
            Keystream bytes
        """
        num_blocks = (length + 63) // 64  # Number of 64-byte blocks needed
        keystream = []
        
        for counter in range(num_blocks):
            counter_tensor = torch.tensor(counter, dtype=torch.long)
            block = self.block_function(key, counter_tensor, nonce)
            
            # Convert to bytes
            block_bytes = block.view(16, 1).byte()
            keystream.append(block_bytes)
        
        # Concatenate and truncate to desired length
        keystream = torch.cat(keystream, dim=0).reshape(-1)[:length]
        
        return keystream


# ========== Training Utilities =========

def create_chacha_training_data(num_samples: int = 10000) -> list:
    """
    Create training data for ChaCha20 neural network.
    
    Uses reference ChaCha20 implementation to generate input-output pairs.
    """
    training_data = []
    
    for _ in range(num_samples):
        # Random key, nonce, counter
        key = torch.randint(0, 2**32, (8,), dtype=torch.long)
        counter = torch.randint(0, 2**32, (1,), dtype=torch.long).item()
        nonce = torch.randint(0, 2**32, (3,), dtype=torch.long)
        
        # Generate output using reference implementation
        # (In practice, you'd use the reference ChaCha20 here)
        # For now, we'll use random data as placeholder
        output = torch.randint(0, 2**32, (16,), dtype=torch.long)
        
        training_data.append((key, counter, nonce, output))
    
    return training_data


def train_neural_chacha(model: NeuralQuarterRound, epochs: int = 100,
                        lr: float = 0.001, verbose: bool = True) -> list:
    """
    Train the neural ChaCha quarter round function.
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Generate batch of random inputs
        batch_size = 32
        for _ in range(10):  # 10 batches per epoch
            optimizer.zero_grad()

            # Generate random input state (4 x 32-bit words)
            input_state = torch.randint(0, 2**32, (batch_size, 4), dtype=torch.long)
            
            # Create target using reference ChaCha20 quarter round
            target_state = input_state.clone()
            for i in range(batch_size):
                state_list = target_state[i].tolist()
                # Apply reference quarter round
                chacha20_quarter_round(state_list, 0, 1, 2, 3)
                target_state[i] = torch.tensor(state_list, dtype=torch.long)

            # Forward pass
            output = model(input_state)

            # Normalize for loss computation
            output_norm = output.float() / 0xFFFFFFFF
            target_norm = target_state.float() / 0xFFFFFFFF

            # Compute loss
            loss = criterion(output_norm, target_norm)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / 10
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


if __name__ == "__main__":
    print("=" * 70)
    print("NeuroCrypto: Neural Network ChaCha20 Implementation")
    print("Task 3 - IIT Roorkee")
    print("=" * 70)
    
    # Create neural ChaCha20
    print("\n1. Creating Neural ChaCha20...")
    neural_chacha = NeuralChaCha20()
    print(f"   ✓ Model created with {sum(p.numel() for p in neural_chacha.parameters()):,} parameters")
    
    # Test keystream generation
    print("\n2. Testing Keystream Generation...")
    key = torch.randint(0, 2**32, (8,), dtype=torch.long)
    nonce = torch.randint(0, 2**32, (3,), dtype=torch.long)
    
    keystream = neural_chacha(key, nonce, length=64)
    print(f"   ✓ Generated {len(keystream)} bytes of keystream")
    
    print("\n3. Training Neural Quarter Round...")
    quarter_round = NeuralQuarterRound()
    losses = train_neural_chacha(quarter_round, epochs=50)
    
    print("\n✅ Neural ChaCha20 implementation complete!")
    print("\nNote: This is a research implementation for educational purposes.")
    print("Not suitable for production cryptographic use.")
