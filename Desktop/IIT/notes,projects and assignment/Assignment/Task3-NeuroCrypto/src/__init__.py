"""
NeuroCrypto: Neural Network Implementations of AES and ChaCha
Task 3 Submission - IIT Roorkee
"""

from .models.neuro_aes import (
    SBoxNeuralNetwork,
    MixColumnsLayer,
    ShiftRowsLayer,
    AddRoundKeyLayer,
    NeuroAES,
    AES_SBOX,
    AES_INV_SBOX,
    train_sbox_net,
    create_sbox_training_data
)

from .models.neuro_chacha import (
    NeuralQuarterRound,
    NeuralChaCha20Block,
    NeuralChaCha20,
    train_neural_chacha
)

__all__ = [
    # Neuro-AES
    'SBoxNeuralNetwork',
    'MixColumnsLayer',
    'ShiftRowsLayer',
    'AddRoundKeyLayer',
    'NeuroAES',
    'AES_SBOX',
    'AES_INV_SBOX',
    'train_sbox_net',
    'create_sbox_training_data',
    
    # Neuro-ChaCha
    'NeuralQuarterRound',
    'NeuralChaCha20Block',
    'NeuralChaCha20',
    'train_neural_chacha',
]

__version__ = '1.0.0'
__author__ = 'Task 3 Submission - IIT Roorkee'
