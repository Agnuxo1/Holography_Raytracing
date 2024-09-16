"""
Hologram-LLM-DEMO2.py: 

Francisco Angulo de Lafuente
11 September 2024
https://github.com/Agnuxo1
https://huggingface.co/Agnuxo
https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3

This script builds upon the basic holographic LLM demonstration 
by incorporating a simplified transformer architecture with attention 
mechanisms. It showcases how embeddings, attention matrices, 
and output projections can be integrated into the holographic framework. 
It also includes a basic training loop for demonstration purposes.
"""

import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft2, ifft2
from numba import cuda
import math

# CUDA device configuration
cuda.select_device(0)

# Simulation parameters
GRID_SIZE = 1024
WAVELENGTH = 532e-9  # Green wavelength (532 nm)
PROPAGATION_DISTANCE = 0.1  # 10 cm
VOCAB_SIZE = 1000
EMBEDDING_DIM = 64
NUM_HEADS = 4
SEQUENCE_LENGTH = 16

# Utility functions
def angular_spectrum_propagation(field, distance, wavelength):
    kx, ky = cp.meshgrid(cp.fft.fftfreq(GRID_SIZE), cp.fft.fftfreq(GRID_SIZE))
    kz = cp.sqrt(1 - (wavelength * kx)**2 - (wavelength * ky)**2)
    return ifft2(fft2(field) * cp.exp(1j * 2 * cp.pi * distance * kz / wavelength))

@cuda.jit
def raytrace_kernel(hologram, neurons, output):
    i, j = cuda.grid(2)
    if i < output.shape[0] and j < output.shape[1]:
        ray_origin = cuda.local.array(3, dtype=cp.float32)
        ray_direction = cuda.local.array(3, dtype=cp.float32)
        
        ray_origin[0] = i / output.shape[0] - 0.5
        ray_origin[1] = j / output.shape[1] - 0.5
        ray_origin[2] = -1.0
        
        ray_direction[0] = 0
        ray_direction[1] = 0
        ray_direction[2] = 1
        
        accumulated_intensity = 0.0
        
        for k in range(neurons.shape[0]):
            sphere_center = neurons[k]
            sphere_radius = 0.01
            
            a = (ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2)
            b = 2 * (ray_direction[0] * (ray_origin[0] - sphere_center[0]) +
                     ray_direction[1] * (ray_origin[1] - sphere_center[1]) +
                     ray_direction[2] * (ray_origin[2] - sphere_center[2]))
            c = ((ray_origin[0] - sphere_center[0])**2 +
                 (ray_origin[1] - sphere_center[1])**2 +
                 (ray_origin[2] - sphere_center[2])**2 - sphere_radius**2)
            
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                t = (-b - math.sqrt(discriminant)) / (2*a)
                if t > 0:
                    intersection_point = (
                        ray_origin[0] + t * ray_direction[0],
                        ray_origin[1] + t * ray_direction[1],
                        ray_origin[2] + t * ray_direction[2]
                    )
                    hologram_value = hologram[
                        int((intersection_point[0] + 0.5) * hologram.shape[0]),
                        int((intersection_point[1] + 0.5) * hologram.shape[1])
                    ]
                    accumulated_intensity += abs(hologram_value)
        
        output[i, j] = accumulated_intensity

def custom_softmax(x, axis=-1):
    x_max = cp.max(x, axis=axis, keepdims=True)
    exp_x = cp.exp(x - x_max)
    return exp_x / cp.sum(exp_x, axis=axis, keepdims=True)

class HolographicLLM:
    def __init__(self, num_neurons, learning_rate=0.01):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        real_part = cp.random.random((GRID_SIZE, GRID_SIZE), dtype=cp.float32)
        imag_part = cp.random.random((GRID_SIZE, GRID_SIZE), dtype=cp.float32)
        self.hologram = real_part + 1j * imag_part
        self.neurons = cp.random.uniform(-0.5, 0.5, (num_neurons, 3))
        
        # Embeddings
        self.embedding_matrix = cp.random.normal(0, 0.1, (VOCAB_SIZE, EMBEDDING_DIM))
        
        # Attention matrices
        self.query_matrix = cp.random.normal(0, 0.1, (EMBEDDING_DIM, EMBEDDING_DIM))
        self.key_matrix = cp.random.normal(0, 0.1, (EMBEDDING_DIM, EMBEDDING_DIM))
        self.value_matrix = cp.random.normal(0, 0.1, (EMBEDDING_DIM, EMBEDDING_DIM))
        
        # Output projection
        self.output_matrix = cp.random.normal(0, 0.1, (EMBEDDING_DIM, VOCAB_SIZE))
    
    def normalize_hologram(self):
        self.hologram /= cp.abs(self.hologram).max()
    
    def propagate_hologram(self):
        self.hologram = angular_spectrum_propagation(
            self.hologram,
            PROPAGATION_DISTANCE,
            WAVELENGTH
        )
        self.normalize_hologram()
    
    def raytrace(self):
        output = cp.zeros((GRID_SIZE, GRID_SIZE), dtype=cp.float32)
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(output.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(output.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        raytrace_kernel[blockspergrid, threadsperblock](
            self.hologram, self.neurons, output
        )
        return output
    
    def attention(self, query, key, value):
        attention_scores = cp.matmul(query, key.T) / cp.sqrt(EMBEDDING_DIM)
        attention_probs = custom_softmax(attention_scores, axis=-1)
        return cp.matmul(attention_probs, value)
    
    def generate_text(self, input_sequence, max_length=50, temperature=0.8):
        generated_sequence = input_sequence.copy()
        
        for _ in range(max_length - len(input_sequence)):
            probabilities = self.inference(cp.array(generated_sequence[-SEQUENCE_LENGTH:]), temperature)
            
            # Ensure probabilities sum to 1
            probabilities = probabilities / cp.sum(probabilities)
            
            # Use random.choice with size=1 and extract the single value
            next_token = cp.random.choice(VOCAB_SIZE, size=1, p=probabilities.get())[0]
            generated_sequence.append(int(next_token))
            
            if next_token == 1:  # Assuming 1 is the end-of-sequence token
                break
        
        return generated_sequence

    def inference(self, input_sequence, temperature=1.0):
        embedded_sequence = self.embedding_matrix[input_sequence]
        query = cp.matmul(embedded_sequence, self.query_matrix)
        key = cp.matmul(embedded_sequence, self.key_matrix)
        value = cp.matmul(embedded_sequence, self.value_matrix)
        
        attention_scores = cp.matmul(query, key.T) / cp.sqrt(EMBEDDING_DIM)
        attention_probs = custom_softmax(attention_scores, axis=-1)
        attended_sequence = cp.matmul(attention_probs, value)
        
        output_logits = cp.matmul(attended_sequence, self.output_matrix)
        output_logits = output_logits.mean(axis=0)  # Average over sequence length
        
        # Apply temperature
        scaled_logits = output_logits / temperature
        
        return custom_softmax(scaled_logits)
    
    def train_step(self, input_sequence, target_sequence):
        # Convert input_sequence and target_sequence to CuPy arrays
        input_sequence = cp.array(input_sequence)
        target_sequence = cp.array(target_sequence)

        # Forward pass
        embedded_sequence = self.embedding_matrix[input_sequence]
        query = cp.matmul(embedded_sequence, self.query_matrix)
        key = cp.matmul(embedded_sequence, self.key_matrix)
        value = cp.matmul(embedded_sequence, self.value_matrix)
        
        attention_scores = cp.matmul(query, key.T) / cp.sqrt(EMBEDDING_DIM)
        attention_probs = custom_softmax(attention_scores, axis=-1)
        attended_sequence = cp.matmul(attention_probs, value)
        
        output_logits = cp.matmul(attended_sequence, self.output_matrix)
        probabilities = custom_softmax(output_logits, axis=-1)
        
        # Compute loss
        loss = -cp.sum(cp.log(probabilities[cp.arange(SEQUENCE_LENGTH), target_sequence]))
        
        # Backpropagation
        dlogits = probabilities.copy()
        dlogits[cp.arange(SEQUENCE_LENGTH), target_sequence] -= 1
        
        # Gradient for output matrix
        doutput_matrix = cp.matmul(attended_sequence.T, dlogits)
        
        # Gradient for attention
        dattended = cp.matmul(dlogits, self.output_matrix.T)
        dvalue = cp.matmul(attention_probs.T, dattended)
        dattention_probs = cp.matmul(dattended, value.T)
        dquery = cp.matmul(dattention_probs, key) / cp.sqrt(EMBEDDING_DIM)
        dkey = cp.matmul(dattention_probs.T, query) / cp.sqrt(EMBEDDING_DIM)
        
        # Gradient for embedding
        dembedding = (cp.matmul(dquery, self.query_matrix.T) + 
                    cp.matmul(dkey, self.key_matrix.T) + 
                    cp.matmul(dvalue, self.value_matrix.T))
        
        # Update parameters
        self.output_matrix -= self.learning_rate * doutput_matrix
        self.query_matrix -= self.learning_rate * cp.matmul(embedded_sequence.T, dquery)
        self.key_matrix -= self.learning_rate * cp.matmul(embedded_sequence.T, dkey)
        self.value_matrix -= self.learning_rate * cp.matmul(embedded_sequence.T, dvalue)
        
        for i, token in enumerate(input_sequence):
            self.embedding_matrix[token] -= self.learning_rate * dembedding[i]
        
        return loss.item()

# Demonstration
if __name__ == "__main__":
    model = HolographicLLM(num_neurons=1000)
    
    # Simple training loop
    for epoch in range(100):
        input_sequence = cp.random.randint(0, VOCAB_SIZE, SEQUENCE_LENGTH).tolist()
        target_sequence = cp.random.randint(0, VOCAB_SIZE, SEQUENCE_LENGTH).tolist()
        loss = model.train_step(input_sequence, target_sequence)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Generate text
    input_sequence = cp.random.randint(0, VOCAB_SIZE, SEQUENCE_LENGTH).tolist()
    print("Input sequence:", input_sequence)
    
    generated_sequence = model.generate_text(input_sequence, temperature=0.8)
    print("Generated sequence:", generated_sequence)
    
    # Perform inference on the complete sequence
    result = model.inference(generated_sequence)
    
    print(f"Output probabilities for the last token: {result}")
    print(f"Most probable token: {cp.argmax(result).item()}")
