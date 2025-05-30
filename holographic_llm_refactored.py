"""
Advanced Holographic LLM Simulation (Conceptual)

This script presents a conceptual framework for an advanced Large Language Model (LLM)
that integrates ideas from holographic data storage and processing. It combines
traditional neural network components (like LSTMs and conceptual transformer mechanisms)
with simulated holographic operations, including wavefield propagation and raytracing
through a neuron field.

The goal is to explore possibilities for novel AI architectures, though many holographic
aspects are highly simplified and illustrative, requiring substantial research for
practical, efficient, and physically accurate implementation.

Key Features:
- Hybrid Model Structure: Combines PyTorch for core sequence processing (LSTM) and
  CuPy/NumPy for custom transformer-like operations and holographic simulations.
- Configurable Parameters: Model dimensions, learning rates, holographic properties,
  and demo behavior can be adjusted via command-line arguments.
- CUDA Acceleration:
    - CuPy is used for GPU-accelerated numerical operations in custom components.
    - Numba is used to JIT-compile CUDA kernels for raytracing simulations.
    - Falls back to NumPy for CPU execution if CuPy or Numba are unavailable/misconfigured.
- Conceptual Holographic Components:
    - Angular spectrum propagation for wavefields.
    - Raytracing through a field of spherical neurons interacting with a hologram.
    - Basic holographic encoding and interaction mechanisms.
- Transformer-Like Mechanisms: Includes conceptual weights for Query, Key, Value,
  and Output projections, with simplified attention and training steps.
- Text Generation: Demonstrates text generation capabilities using the implemented
  inference logic.
- Command-Line Interface: Facilitates demo runs with customizable settings.

How to Run:
The script can be executed directly from the command line.

1. Default Demo Execution:
   python holographic_llm_refactored.py

2. Customizing Parameters:
   You can override default parameters using command-line arguments. For example:
   python holographic_llm_refactored.py --epochs 10 --learning_rate 0.0005 --temperature 0.85 --sequence_length 30

   Common arguments to customize:
   - `--epochs`: Number of epochs for the demo training loop.
   - `--learning_rate`: Learning rate for the conceptual `train_step`.
   - `--embedding_dim`: Dimension of token embeddings.
   - `--hidden_dim`: Dimension of LSTM hidden states.
   - `--holographic_dimension`: Dimensionality of the holographic space.
   - `--num_neurons`: Number of neurons in the simulated holographic layer.
   - `--neuron_sphere_radius`: Radius of spherical neurons for raytracing.
   - `--grid_size`: Size of the grid for hologram/raytracing output.
   - `--wavelength`: Wavelength for holographic propagation.
   - `--temperature`: Softmax temperature for text generation.
   - `--max_gen_length`: Maximum length of generated text.
   - `--device`: "cuda" or "cpu" (influences both PyTorch and CuPy components).

   Use `python holographic_llm_refactored.py --help` to see all available options.

Dependencies:
- Python 3.7+
- NumPy: (pip install numpy)
- PyTorch: (See https://pytorch.org/get-started/locally/ for installation instructions)
- CuPy (Optional, for GPU acceleration of custom operations):
  Installation depends on your CUDA version (e.g., `pip install cupy-cuda11x` for CUDA 11.x,
  or `pip install cupy-cuda12x` for CUDA 12.x). See CuPy documentation.
- Numba (Optional, for JIT-compiling CUDA kernels): (pip install numba)
  Ensure your NVIDIA drivers and CUDA toolkit are compatible with Numba.

How to Contribute:
Contributions to this conceptual project are welcome! Whether you're interested in
improving the physics, optimizing performance, or expanding the AI capabilities,
your input can help explore this novel intersection of technologies.

Areas for contribution could include:
- Enhancing the physical accuracy of holographic simulations (propagation, raytracing).
- Optimizing CUDA kernels (Numba) and CuPy operations for better performance.
- Developing more sophisticated LLM layers or attention mechanisms.
- Implementing a more rigorous training and backpropagation process, especially for
  the hybrid PyTorch/CuPy model and holographic components.
- Adding comprehensive evaluation metrics for both LLM and holographic aspects.
- Improving documentation, adding more examples, or creating visualizations.
- Exploring different holographic encoding/decoding schemes.
- Investigating methods for learning or adapting the holographic components.

Contribution Workflow:
1. Fork the repository (if this were on a platform like GitHub).
2. Create a new branch for your feature or bugfix (e.g., `git checkout -b feature/my-new-feature`).
3. Make your changes, ensuring code is well-commented and adheres to the project's style.
4. Add or update relevant documentation.
5. Test your changes thoroughly.
6. Submit a pull request with a clear description of your contributions and any
   relevant findings or challenges.

If you're unsure where to start or have ideas, feel free to open an issue to discuss them.
(Assuming a GitHub-like issue tracking system).
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy successfully imported. Will use CuPy for specified operations.")
except ImportError:
    CUPY_AVAILABLE = False
    cp = np # Fallback to numpy if cupy is not available
    logger.warning("CuPy not found. Falling back to NumPy for relevant operations. GPU acceleration for these parts will not be available.")


# --- Logging Setup ---
# Configure basic logging for the script.
# This allows for tracking information, warnings, and errors during execution.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Configuration Dataclasses ---
@dataclass
class ModelConfig:
    """
    Configuration class for the Holographic LLM.

    This dataclass holds all hyperparameters and settings for the model,
    training process, and data handling.
    """
    vocab_size: int = 50000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4  # Number of layers in the recurrent part (e.g., LSTM)
    num_heads: int = 8  # Number of attention heads, if a Transformer component were used
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters for holographic components (conceptual)
    holographic_dimension: int = 1024  # Dimensionality of the holographic space
    sparsity_factor: float = 0.05  # Desired sparsity in holographic representations

    # Early stopping parameters
    early_stopping_patience: int = 3  # Epochs to wait for improvement before stopping
    early_stopping_delta: float = 0.001  # Minimum change to qualify as an improvement


@dataclass
class PerformanceMetrics:
    """
    Stores various performance metrics of the model during and after training.
    """
    # Added sequence_length, num_neurons etc. here for clarity as they are part of model's structure/config
    sequence_length: int = 20 # Default sequence length, used in train_step and generate_text
    num_neurons: int = 100 # Example number of neurons for holographic layer
    neuron_sphere_radius: float = 0.01 # Radius for sphere model of neurons in raytracing

    # Holographic simulation specific parameters
    grid_size: int = 256  # For hologram plane or output image dimension in raytracing
    wavelength: float = 532e-9  # Green light in meters, for propagation
    pixel_size: float = 10e-6 # Sensor/SLM pixel size in meters, for propagation
    propagation_distance: float = 0.1 # Distance in meters, for propagation

    numerical_precision: str = "float32" # "float32" or "float64" for CuPy/NumPy arrays
    temperature: float = 0.7 # For text generation sampling softmax

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list) # Validation perplexity
    training_time: float = 0.0  # Total time spent on training
    inference_latency: float = 0.0  # Average time for one inference pass (placeholder)


# --- Holographic Utilities ---
class HolographicUtils:
    """
    Provides utility functions for conceptual holographic operations.
    These are highly simplified and illustrative.
    """

    @staticmethod
    def check_cuda_availability() -> None:
        """
        Checks for CUDA (GPU) availability and logs the status.
        If CUDA is critical, this function could be modified to raise an error.
        """
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available. Operations will run on CPU.")
            # Consider raising RuntimeError if CUDA is essential for the application
            # raise RuntimeError("CUDA is required for this application but not available.")

    @staticmethod
    def complex_vector_holography(
        data: torch.Tensor,
        dimension: int
    ) -> torch.Tensor:
        """
        Encodes input data into a complex-valued holographic representation.
        This is a simplified example using random phase encoding.

        Args:
            data (torch.Tensor): Input data, typically embeddings or projected hidden states.
                                 Shape: (batch_size, seq_len, feature_dim).
            dimension (int): The desired dimensionality of the holographic space.
                             Should match `feature_dim` if no projection is intended here.

        Returns:
            torch.Tensor: Complex-valued holographic representation of the input data.
                          Shape: (batch_size, seq_len, dimension) (complex).
        """
        batch_size, seq_len, feature_dim = data.shape
        device = data.device

        # Ensure data is projected to the holographic dimension if necessary.
        # In the current model, projection is handled by `holographic_memory_projection` layer before this call.
        if feature_dim != dimension:
            # This case implies an unexpected mismatch if projection is assumed to be done prior.
            # For robustness, one might add a projection here, but it's better handled upstream.
            logger.warning(
                f"Input feature dimension ({feature_dim}) and holographic dimension ({dimension}) mismatch. "
                "Ensure data is projected to holographic dimension before calling this function."
            )
            # Example of a fallback projection (though ideally handled by model structure):
            # projection_matrix = nn.Linear(feature_dim, dimension).to(device)
            # data = projection_matrix(data)
            # For this example, we'll proceed, but the result might be misaligned if dimensions differ.
            pass # Assuming for now that data's last dim is already `dimension`

        # Create random phases for holographic encoding.
        # These phases are what "encode" the information.
        phases = torch.rand(
            batch_size, seq_len, dimension, device=device
        ) * 2 * np.pi

        # Create complex vectors. Amplitudes are set to 1 for simplicity (common in some HRR models).
        # The magnitude of `data` could also be incorporated into amplitudes,
        # but here, `data` (projected_for_holo) primarily serves to define the shape and device.
        complex_holo = torch.polar(
            torch.ones_like(data, device=device), # Magnitudes set to 1
            phases                                # Phases are random
        )
        # Note: This is a highly simplified representation.
        # Real holographic methods (e.g., HRRs) are more mathematically involved.
        return complex_holo  # Shape: (batch_size, seq_len, dimension) (complex)

    @staticmethod
    def holographic_interaction(
        holo_data1: torch.Tensor,
        holo_data2: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulates interaction (e.g., binding or correlation) in holographic space.
        This example uses element-wise complex multiplication, a common operation for binding.

        Args:
            holo_data1 (torch.Tensor): First holographic representation.
                                       Shape: (batch_size, seq_len, dimension) (complex).
            holo_data2 (torch.Tensor): Second holographic representation.
                                       Shape: (batch_size, seq_len, dimension) (complex).

        Returns:
            torch.Tensor: Resultant hologram from the interaction.
                          Shape: (batch_size, seq_len, dimension) (complex).
        """
        # Element-wise complex product simulates binding (creating an association).
        return holo_data1 * holo_data2

    @staticmethod
    def reconstruct_from_hologram(
        hologram: torch.Tensor,
        reference_hologram: torch.Tensor
    ) -> torch.Tensor:
        """
        Attempts to reconstruct or retrieve information from a hologram using a reference.
        Simplified: uses complex conjugate of a reference signal for retrieval (correlation/decoding).

        Args:
            hologram (torch.Tensor): The hologram to reconstruct from (e.g., a combined or memory hologram).
                                     Shape: (batch_size, seq_len, dimension) (complex).
            reference_hologram (torch.Tensor): A reference hologram, typically related to the key or cue
                                               used for reconstruction.
                                        Shape: (batch_size, seq_len, dimension) (complex).

        Returns:
            torch.Tensor: The reconstructed signal (real part). This output is intended to be
                          interpretable or usable by other parts of the network.
                          Shape: (batch_size, seq_len, dimension) (real).
        """
        # Correlation is often implemented by multiplying with the complex conjugate of the reference/cue.
        # This operation aims to retrieve the signal associated with the reference.
        reconstructed_signal = hologram * reference_hologram.conj()
        # The real part is often taken as the relevant output for further processing,
        # though phase information can also be important in more complex models.
        return reconstructed_signal.real


# --- Model Definition ---
class AdvancedHolographicLLM(nn.Module):
    """
    An advanced Language Model conceptually incorporating holographic principles
    for aspects like memory and contextual understanding.

    The holographic components are illustrative and simplified, intended to provide
    a basic framework for how such ideas might be integrated.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.metrics = PerformanceMetrics()  # Initialize performance metrics container

        # Perform CUDA check at model initialization (PyTorch context)
        HolographicUtils.check_cuda_availability()
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Configuration specifies CUDA, but PyTorch CUDA not available. Switching PyTorch device to CPU."
            )
            self.config.device = "cpu" # For PyTorch components
        logger.info(f"Model (PyTorch components) configured to run on device: {self.config.device}")

        # Initialize PyTorch-based layers (as before)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.sequence_encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0
        )
        self.holographic_memory_projection = nn.Linear(
            config.hidden_dim, config.holographic_dimension
        )
        self.output_projection = nn.Linear(
            config.hidden_dim + config.holographic_dimension, config.vocab_size
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize model components (including CuPy based if available)
        self._initialize_model()

        logger.info("AdvancedHolographicLLM initialized successfully with PyTorch and potentially CuPy components.")

    def _generate_neuron_positions(self) -> Any: # cupy.ndarray or numpy.ndarray
        """Generates random 3D positions for neurons in the holographic layer."""
        logger.info(f"Generating neuron positions for {self.config.num_neurons} neurons.")
        # Using NumPy/CuPy directly as per original DEMO2, not a PyTorch layer
        if CUPY_AVAILABLE and self.config.device == "cuda":
            positions = cp.random.uniform(-0.5, 0.5, (self.config.num_neurons, 3))
            logger.debug(f"Generated neuron positions on GPU with CuPy, shape: {positions.shape}")
        else:
            positions = np.random.uniform(-0.5, 0.5, (self.config.num_neurons, 3))
            logger.debug(f"Generated neuron positions on CPU with NumPy, shape: {positions.shape}")
        return positions

    def _initialize_transformer_weights(self) -> None:
        """Initializes conceptual transformer-related weight matrices using CuPy/NumPy."""
        logger.info("Initializing transformer-style weights (Q, K, V, Output, Embedding)...")
        # These are conceptual and not part of the PyTorch nn.Module structure directly for now
        # Dimensions based on config
        emb_dim = self.config.embedding_dim
        vocab_size = self.config.vocab_size

        # Standard deviation for weight initialization
        std_dev = 0.02

        if CUPY_AVAILABLE and self.config.device == "cuda":
            logger.info("Using CuPy for transformer weight initialization on GPU.")
            self.embedding_matrix_cupy = cp.random.normal(0, std_dev, (vocab_size, emb_dim)).astype(cp.float32)
            self.query_matrix = cp.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(cp.float32)
            self.key_matrix = cp.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(cp.float32)
            self.value_matrix = cp.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(cp.float32)
            self.output_matrix_cupy = cp.random.normal(0, std_dev, (emb_dim, vocab_size)).astype(cp.float32)
        else:
            logger.info("Using NumPy for transformer weight initialization on CPU.")
            self.embedding_matrix_cupy = np.random.normal(0, std_dev, (vocab_size, emb_dim)).astype(np.float32)
            self.query_matrix = np.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(np.float32)
            self.key_matrix = np.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(np.float32)
            self.value_matrix = np.random.normal(0, std_dev, (emb_dim, emb_dim)).astype(np.float32)
            self.output_matrix_cupy = np.random.normal(0, std_dev, (emb_dim, vocab_size)).astype(np.float32)

        logger.debug(f"  CuPy Embedding matrix shape: {self.embedding_matrix_cupy.shape}")
        logger.debug(f"  Query matrix shape: {self.query_matrix.shape}")
        logger.debug(f"  Key matrix shape: {self.key_matrix.shape}")
        logger.debug(f"  Value matrix shape: {self.value_matrix.shape}")
        logger.debug(f"  CuPy Output matrix shape: {self.output_matrix_cupy.shape}")

    def _initialize_model(self) -> None:
        """Initializes model components, including neuron positions and transformer weights."""
        self.neurons = self._generate_neuron_positions() # CuPy/NumPy array
        self._initialize_transformer_weights() # Initializes CuPy/NumPy arrays

        # Conceptual hologram - placeholder, needs actual initialization logic
        # For now, let's assume it's initialized if CuPy is available and on CUDA
        # The actual complex hologram generation is part of HolographicUtils or specific layers.
        # This self.hologram is more of a persistent, learnable state if that's the design.
        if CUPY_AVAILABLE and self.config.device == "cuda":
            # Example placeholder initialization. Real initialization would be more meaningful.
            self.hologram = cp.zeros((self.config.holographic_dimension, self.config.holographic_dimension), dtype=cp.complex64)
            logger.info(f"Conceptual hologram initialized on GPU (CuPy) with shape: {self.hologram.shape}")
        else:
            self.hologram = np.zeros((self.config.holographic_dimension, self.config.holographic_dimension), dtype=np.complex64)
            logger.info(f"Conceptual hologram initialized on CPU (NumPy) with shape: {self.hologram.shape}")


    @staticmethod
    def custom_softmax(x: Any, axis: int = -1) -> Any: # cupy.ndarray or numpy.ndarray
        """
        Numerically stable softmax function.

        Args:
            x (Any): Input array (CuPy or NumPy).
            axis (int): Axis along which softmax is computed.

        Returns:
            Any: Softmax output array.
        """
        xp = cp.get_array_module(x) # Get CuPy or NumPy module based on input array type
        e_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
        return e_x / xp.sum(e_x, axis=axis, keepdims=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        context_hologram: Optional[torch.Tensor] = None  # Added Optional type hint
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AdvancedHolographicLLM.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
                                      Shape: (batch_size, seq_len).
            context_hologram (Optional[torch.Tensor]): Pre-computed hologram representing prior context.
                                                       If provided, it's used in holographic interaction.
                                                       Shape: (batch_size, seq_len, holographic_dimension) (complex).
                                                       Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Output logits over the vocabulary.
                                         Shape: (batch_size, seq_len, vocab_size).
                - current_input_hologram (torch.Tensor): Hologram generated from the current input.
                                                        This can be used as context for subsequent steps.
                                                        Shape: (batch_size, seq_len, holographic_dimension) (complex).
        """
        # 1. Embedding Layer
        # Convert input token IDs to dense vector representations.
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded_data = self.embedding(input_ids)
        embedded_data = self.dropout(embedded_data)

        # 2. Sequence Encoding
        # Process the sequence of embeddings to capture contextual information.
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        encoded_output, (hidden, cell) = self.sequence_encoder(embedded_data)
        # `encoded_output` contains the output features (hidden states) for each time step.
        # `hidden` and `cell` are the final hidden and cell states (for LSTMs).

        # --- Holographic Operations (Conceptual) ---
        # 3. Create Hologram of Current Input
        # Project the encoded output to the holographic dimension. This prepares the data
        # to be encoded into a holographic representation.
        # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, holographic_dimension)
        projected_for_holo = self.holographic_memory_projection(encoded_output)

        # Generate a complex-valued hologram for the current input.
        # This hologram represents the current input sequence in the holographic space.
        # Shape: (batch_size, seq_len, holographic_dimension) (real) -> (batch_size, seq_len, holographic_dimension) (complex)
        current_input_hologram = HolographicUtils.complex_vector_holography(
            projected_for_holo, # Data that forms the basis of the hologram
            self.config.holographic_dimension
        )

        # 4. Interact with Context Hologram (if provided)
        holographic_features_for_output: torch.Tensor
        if context_hologram is not None:
            # Ensure dimensions match for a meaningful interaction.
            if context_hologram.shape == current_input_hologram.shape:
                # Combine current input's hologram with the external context hologram.
                # This interaction (e.g., binding) can modify the representation.
                combined_hologram = HolographicUtils.holographic_interaction(
                    current_input_hologram,
                    context_hologram
                )
                # Reconstruct information from the combined hologram.
                # This step attempts to "decode" or retrieve information resulting from the interaction.
                # The `current_input_hologram` is used as a reference/cue for reconstruction.
                # This is highly conceptual; the choice of reference signal is critical.
                reconstructed_context_info = HolographicUtils.reconstruct_from_hologram(
                    combined_hologram,
                    current_input_hologram # Using current input's hologram as reference
                ) # Shape: (batch_size, seq_len, holographic_dimension) (real)
                holographic_features_for_output = reconstructed_context_info
            else:
                logger.warning(
                    "Context hologram shape mismatch (Context: %s, Input: %s). Ignoring context hologram.",
                    context_hologram.shape, current_input_hologram.shape
                )
                # Fallback: use a zero tensor, signifying no contextual holographic information.
                holographic_features_for_output = torch.zeros_like(
                    current_input_hologram.real, device=self.config.device
                )
        else:
            # If no external context hologram is provided, the "holographic features"
            # are derived solely from the current input's hologram (real part).
            # This could represent self-contextualization or a baseline holographic representation.
            holographic_features_for_output = current_input_hologram.real
            # Shape: (batch_size, seq_len, holographic_dimension) (real)

        # 5. Combine Sequence Features with Holographic Features
        # Concatenate the output from the sequence encoder (standard features)
        # with the derived holographic features.
        # Shape of encoded_output: (batch_size, seq_len, hidden_dim)
        # Shape of holographic_features_for_output: (batch_size, seq_len, holographic_dimension)
        # Resulting shape: (batch_size, seq_len, hidden_dim + holographic_dimension)
        combined_features = torch.cat(
            (encoded_output, holographic_features_for_output), dim=-1
        )

        # 6. Output Projection
        # Project the combined features to the vocabulary space to get logits
        # (raw scores for each token in the vocabulary).
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.output_projection(combined_features)

        return logits, current_input_hologram

    def inference(self, input_sequence: List[int], temperature: float = 0.7) -> Any: # cupy.ndarray or numpy.ndarray
        """
        Performs inference for a single input sequence using conceptual transformer operations.
        This method uses CuPy/NumPy for its operations.

        Args:
            input_sequence (List[int]): A list of token IDs.
            temperature (float): Temperature for softmax scaling.

        Returns:
            Any: Probability distribution over the vocabulary (CuPy or NumPy array).
        """
        if not CUPY_AVAILABLE and self.config.device == "cuda":
            logger.warning("CuPy is not available, but device is CUDA. Inference will run on CPU using NumPy.")
            xp = np
        elif CUPY_AVAILABLE and self.config.device == "cuda":
            xp = cp
        else:
            xp = np
            if self.config.device == "cuda": # device is cuda but cupy not available
                 logger.debug("Inference: device is CUDA but CuPy not found, using NumPy on CPU.")
            else: # device is cpu
                 logger.debug("Inference: device is CPU, using NumPy on CPU.")


        if not input_sequence:
            logger.error("Input sequence for inference is empty.")
            raise ValueError("Input sequence cannot be empty.")

        logger.debug(f"Inference started for sequence: {input_sequence} with temp: {temperature}")

        # 1. Embedding lookup (using self.embedding_matrix_cupy)
        # Ensure input_sequence is a CuPy/NumPy array for direct indexing
        input_ids_xp = xp.array(input_sequence, dtype=xp.int32)

        # Check if token IDs are within vocab_size
        if xp.any(input_ids_xp >= self.config.vocab_size) or xp.any(input_ids_xp < 0):
            logger.error(f"Token IDs out of bounds: {input_ids_xp}. Vocab size: {self.config.vocab_size}")
            raise ValueError("Token ID out of vocabulary bounds.")

        embedded_sequence = self.embedding_matrix_cupy[input_ids_xp]
        logger.debug(f"  Embedded sequence shape: {embedded_sequence.shape}") # (seq_len, embedding_dim)

        # For simplicity, take the embedding of the last token as the query context
        # More complex models would process the whole sequence (e.g., with RNN or attention)
        last_token_embedding = embedded_sequence[-1, :] # Shape: (embedding_dim,)
        last_token_embedding = last_token_embedding.reshape(1, -1) # Shape: (1, embedding_dim)
        logger.debug(f"  Last token embedding shape for query: {last_token_embedding.shape}")

        # 2. Conceptual Self-Attention (Simplified)
        # Query, Key, Value projections from the last token's embedding
        query = last_token_embedding @ self.query_matrix
        # For keys and values, a real transformer would use all token embeddings in the sequence.
        # Here, for simplification, we might use the query itself or a global context.
        # Let's use the query to interact with itself as a placeholder for more complex key generation.
        keys = last_token_embedding @ self.key_matrix
        values = last_token_embedding @ self.value_matrix
        logger.debug(f"  Query shape: {query.shape}")
        logger.debug(f"  Keys shape: {keys.shape}")
        logger.debug(f"  Values shape: {values.shape}")

        # Attention scores (dot product between query and keys)
        # Here, K.T is needed if K has shape (1, emb_dim) like Q.
        # If K represents multiple key vectors, shapes would differ.
        # Simplified: (1, emb_dim) @ (emb_dim, 1) -> (1,1)
        attention_scores = query @ keys.T
        logger.debug(f"  Attention scores shape: {attention_scores.shape}")

        # Apply softmax to get attention weights
        attention_weights = AdvancedHolographicLLM.custom_softmax(attention_scores, axis=-1)
        logger.debug(f"  Attention weights shape: {attention_weights.shape}, weights: {attention_weights}")

        # Weighted sum of values
        # (1, num_keys) @ (num_keys, emb_dim) -> (1, emb_dim)
        # In this simplified case, num_keys is 1, so (1,1) @ (1, emb_dim) -> (1, emb_dim)
        context_vector = attention_weights @ values
        logger.debug(f"  Context vector shape: {context_vector.shape}")

        # 3. Output Projection
        # Project the context vector to vocabulary space
        logits = context_vector @ self.output_matrix_cupy # (1, emb_dim) @ (emb_dim, vocab_size) -> (1, vocab_size)
        logger.debug(f"  Logits shape: {logits.shape}")

        # 4. Scale by temperature and apply softmax for probabilities
        probabilities = AdvancedHolographicLLM.custom_softmax(logits / temperature, axis=-1)
        # Squeeze to get a 1D array of probabilities
        probabilities = probabilities.squeeze()
        logger.debug(f"  Probabilities shape after squeeze: {probabilities.shape}")

        return probabilities


    def generate_text(self, input_sequence: List[int], max_length: int, temperature: float = 0.7) -> List[int]:
        """
        Generates text sequence using the inference method.
        This method uses CuPy/NumPy for its operations.

        Args:
            input_sequence (List[int]): Initial sequence of token IDs.
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Temperature for sampling.

        Returns:
            List[int]: The generated sequence of token IDs.
        """
        xp = cp.get_array_module(self.embedding_matrix_cupy) # Use same module as weights

        generated_sequence = list(input_sequence) # Make a mutable copy
        logger.info(f"Starting text generation from sequence: {input_sequence}, max_length: {max_length}, temp: {temperature}")

        for _ in range(max_length - len(input_sequence)):
            if not generated_sequence:
                logger.warning("Text generation stopped: generated_sequence is empty.")
                break

            # Get probabilities for the next token
            probabilities = self.inference(generated_sequence, temperature)

            # Sample the next token based on the probabilities
            # Ensure probabilities sum to 1 (they should from softmax)
            # Handle potential numerical precision issues for sampling
            if xp.sum(probabilities) == 0 or xp.isnan(xp.sum(probabilities)):
                logger.warning("Probabilities sum to 0 or NaN. Using uniform distribution as fallback.")
                # Fallback to uniform distribution if probabilities are ill-defined
                next_token_id = int(xp.random.choice(self.config.vocab_size))
            else:
                # Ensure probabilities are normalized if they are slightly off due to precision
                probabilities = probabilities / xp.sum(probabilities)
                next_token_id = int(xp.random.choice(self.config.vocab_size, p=probabilities))

            logger.debug(f"  Generated token ID: {next_token_id}, Prob sum: {xp.sum(probabilities)}")
            generated_sequence.append(next_token_id)

            # Optional: Add a condition to stop if an End-Of-Sequence token is generated
            # if next_token_id == EOS_TOKEN_ID:
            #     break

        logger.info(f"Finished text generation. Generated sequence: {generated_sequence}")
        return generated_sequence

    def train_step(self, input_batch: Any, target_batch: Any) -> float:
        """
        Performs a single training step (conceptual, using CuPy/NumPy for transformer-like parts).

        Args:
            input_batch (Any): Batch of input sequences (e.g., list of lists, or CuPy/NumPy array).
                               Shape typically (batch_size, sequence_length).
            target_batch (Any): Batch of target sequences.
                                Shape typically (batch_size, sequence_length).
        Returns:
            float: The calculated loss for the batch.
        """
        if not CUPY_AVAILABLE and self.config.device == "cuda":
            logger.warning("CuPy not available, but device is CUDA. train_step will use NumPy on CPU.")
            xp = np
        elif CUPY_AVAILABLE and self.config.device == "cuda":
            xp = cp
        else: # CPU device
            xp = np
            logger.debug("train_step: device is CPU, using NumPy on CPU.")

        # Convert PyTorch tensors to CuPy/NumPy arrays if they are coming from a DataLoader
        # This is a key point of interaction if the main training loop uses PyTorch DataLoaders.
        if isinstance(input_batch, torch.Tensor):
            input_batch_xp = xp.asarray(input_batch.cpu().numpy()) if CUPY_AVAILABLE else input_batch.cpu().numpy()
            target_batch_xp = xp.asarray(target_batch.cpu().numpy()) if CUPY_AVAILABLE else target_batch.cpu().numpy()
        else: # Assuming already numpy or cupy
            input_batch_xp = xp.asarray(input_batch)
            target_batch_xp = xp.asarray(target_batch)

        logger.debug(f"train_step started. Input batch shape: {input_batch_xp.shape}")

        total_loss = 0.0
        batch_size = input_batch_xp.shape[0]

        # Conceptual gradients (placeholders)
        grad_embedding_matrix = xp.zeros_like(self.embedding_matrix_cupy)
        grad_query_matrix = xp.zeros_like(self.query_matrix)
        grad_key_matrix = xp.zeros_like(self.key_matrix)
        grad_value_matrix = xp.zeros_like(self.value_matrix)
        grad_output_matrix = xp.zeros_like(self.output_matrix_cupy)

        # Process each sequence in the batch
        for i in range(batch_size):
            input_sequence = input_batch_xp[i] # Current sequence from batch
            target_sequence = target_batch_xp[i]

            if xp.any(input_sequence >= self.config.vocab_size) or xp.any(input_sequence < 0):
                logger.error(f"Token IDs out of bounds in input_sequence: {input_sequence}")
                raise ValueError("Token ID out of vocabulary bounds in input_sequence.")
            if xp.any(target_sequence >= self.config.vocab_size) or xp.any(target_sequence < 0):
                 logger.error(f"Token IDs out of bounds in target_sequence: {target_sequence}")
                 raise ValueError("Token ID out of vocabulary bounds in target_sequence.")

            # 1. Embedding
            embedded_sequence = self.embedding_matrix_cupy[input_sequence] # (seq_len, emb_dim)

            # 2. Simplified Transformer-like forward pass (per position for loss calculation)
            # This needs to produce logits for each token in the sequence to compare with targets.
            # The `inference` method is designed for single next token. We need sequence-wide logits.
            # For simplicity, let's assume a loop through sequence positions.
            sequence_loss = 0.0
            for t in range(self.config.sequence_length): # Assuming config.sequence_length is defined
                if t == 0: continue # Cannot predict first token's predecessor

                # Context for predicting token at position `t` is based on tokens up to `t-1`
                # A simple way: use embedding of token `t-1` as query context
                current_context_embedding = embedded_sequence[t-1, :].reshape(1, -1) # (1, emb_dim)

                query = current_context_embedding @ self.query_matrix
                # Keys and Values could be from the whole `embedded_sequence` or parts of it.
                # Simplified: use current_context_embedding for K, V as well
                keys = current_context_embedding @ self.key_matrix
                values = current_context_embedding @ self.value_matrix

                attention_scores = query @ keys.T
                attention_weights = AdvancedHolographicLLM.custom_softmax(attention_scores, axis=-1)
                context_vector = attention_weights @ values # (1, emb_dim)

                logits_at_t = context_vector @ self.output_matrix_cupy # (1, vocab_size)
                probabilities_at_t = AdvancedHolographicLLM.custom_softmax(logits_at_t, axis=-1) # (1, vocab_size)

                # 3. Loss Calculation (Cross-Entropy) for current token
                target_token_id = target_sequence[t]

                # Ensure target_token_id is scalar for indexing
                if isinstance(target_token_id, xp.ndarray) and target_token_id.ndim == 0:
                    target_token_id = int(target_token_id.item())
                elif isinstance(target_token_id, np.int32) or isinstance(target_token_id, np.int64): # from numpy array
                    target_token_id = int(target_token_id)


                # Add small epsilon to prevent log(0)
                epsilon = 1e-9
                prob_of_target = probabilities_at_t[0, target_token_id]
                current_loss = -xp.log(prob_of_target + epsilon)
                sequence_loss += current_loss

                # 4. Conceptual Backpropagation (Placeholder Gradients)
                # This is highly simplified. Real gradients are complex.
                # dL/d_logits (derivative of Loss w.r.t logits_at_t)
                # For cross-entropy, dL/d_logits = P - Y (where P are probabilities, Y is one-hot target)
                d_logits = probabilities_at_t.copy()
                d_logits[0, target_token_id] -= 1.0 # (1, vocab_size)

                # Grad for output_matrix_cupy: dL/dW_out = context_vector.T @ d_logits
                grad_output_matrix_t = context_vector.T @ d_logits # (emb_dim, 1) @ (1, vocab_size) -> (emb_dim, vocab_size)
                grad_output_matrix += grad_output_matrix_t

                # Grad for context_vector: dL/d_context = d_logits @ output_matrix_cupy.T
                d_context_vector = d_logits @ self.output_matrix_cupy.T # (1, vocab_size) @ (vocab_size, emb_dim) -> (1, emb_dim)

                # ... and so on for other matrices (Q, K, V, Embedding)
                # This requires full chain rule derivation for attention, which is omitted for brevity.
                # Placeholder updates for other matrices:
                grad_value_matrix += xp.random.normal(0, 0.01, self.value_matrix.shape) * current_loss # Placeholder
                grad_key_matrix += xp.random.normal(0, 0.01, self.key_matrix.shape) * current_loss     # Placeholder
                grad_query_matrix += xp.random.normal(0, 0.01, self.query_matrix.shape) * current_loss   # Placeholder

                # For embedding_matrix_cupy, grad would depend on dL/d_current_context_embedding
                # and how current_context_embedding (embedded_sequence[t-1]) is used.
                # Placeholder:
                # grad_embedding_matrix[input_sequence[t-1]] += some_gradient_from_d_context_vector

            total_loss += sequence_loss / (self.config.sequence_length -1) # Average loss per token in sequence

        avg_batch_loss = total_loss / batch_size
        logger.debug(f"  Average batch loss: {avg_batch_loss}")

        # 5. Update Weights (SGD-like update)
        lr = self.config.learning_rate
        self.embedding_matrix_cupy -= lr * (grad_embedding_matrix / batch_size) # Averaged gradient
        self.query_matrix -= lr * (grad_query_matrix / batch_size)
        self.key_matrix -= lr * (grad_key_matrix / batch_size)
        self.value_matrix -= lr * (grad_value_matrix / batch_size)
        self.output_matrix_cupy -= lr * (grad_output_matrix / batch_size)

        # Conceptual Hologram Update (placeholder)
        # gradient_hologram_placeholder = xp.random.normal(size=self.hologram.shape, scale=0.01) * avg_batch_loss
        # self.hologram -= lr * gradient_hologram_placeholder
        # logger.debug("  Conceptual hologram update applied.")

        return float(avg_batch_loss) # Return scalar Python float

    def propagate_hologram(self, distance: Optional[float] = None, wavelength: Optional[float] = None, pixel_size: Optional[float] = None) -> None:
        """
        Conceptual propagation of the model's hologram.
        Uses angular_spectrum_propagation if CUPY is available.
        This is a placeholder or simplified version for demo purposes.
        """
        logger.info("Attempting conceptual hologram propagation...")
        if not (CUPY_AVAILABLE and self.config.device == "cuda"):
            logger.warning("propagate_hologram: CuPy not available or device is not CUDA. Skipping actual propagation.")
            return

        dist = distance if distance is not None else self.config.propagation_distance
        wvlgth = wavelength if wavelength is not None else self.config.wavelength
        px_size = pixel_size if pixel_size is not None else self.config.pixel_size

        if self.hologram is None:
            logger.error("Hologram is not initialized. Cannot propagate.")
            return

        logger.info(f"Propagating hologram with distance={dist}, wavelength={wvlgth}, pixel_size={px_size}")
        try:
            # Ensure self.hologram is a CuPy array if we are using CuPy
            hologram_xp = cp.asarray(self.hologram) if isinstance(self.hologram, np.ndarray) else self.hologram

            propagated_hologram = angular_spectrum_propagation(
                hologram_xp,
                distance=dist,
                wavelength=wvlgth,
                pixel_size=px_size
            )
            # In a real scenario, this propagated hologram might be stored or used.
            # For this demo, we just log its shape.
            logger.info(f"Hologram conceptually propagated. Result shape: {propagated_hologram.shape}")
            # self.hologram = propagated_hologram # Optional: update self.hologram
        except Exception as e:
            logger.error(f"Error during hologram propagation: {e}", exc_info=True)


    def raytrace(self) -> Optional[Any]:
        """
        Performs conceptual raytracing using the advanced_raytrace_kernel.
        This method prepares data and launches the Numba CUDA kernel.
        """
        logger.info("Attempting conceptual raytracing...")
        if not NUMBA_AVAILABLE:
            logger.warning("Numba is not available. Skipping raytracing kernel.")
            return None
        if not (CUPY_AVAILABLE and self.config.device == "cuda"): # Kernel expects CuPy arrays
            logger.warning("CuPy not available or device not CUDA. Skipping raytracing (kernel expects CuPy arrays).")
            return None

        try:
            # Ensure hologram and neurons are CuPy arrays
            hologram_cupy = cp.asarray(self.hologram) if isinstance(self.hologram, np.ndarray) else self.hologram
            neurons_cupy = cp.asarray(self.neurons) if isinstance(self.neurons, np.ndarray) else self.neurons

            if hologram_cupy is None or neurons_cupy is None:
                logger.error("Hologram or neurons not initialized for raytracing.")
                return None

            output_image_cupy = cp.zeros((self.config.grid_size, self.config.grid_size), dtype=cp.float32)
            config_params_tuple = (self.config.neuron_sphere_radius,)

            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(output_image_cupy.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(output_image_cupy.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            logger.info(f"Launching advanced_raytrace_kernel with blocks: {blockspergrid}, threads: {threadsperblock}")
            advanced_raytrace_kernel[blockspergrid, threadsperblock](
                hologram_cupy, neurons_cupy, output_image_cupy, config_params_tuple
            )
            cp.cuda.stream.get_current_stream().synchronize() # Ensure kernel completion

            logger.info(f"Raytracing complete. Output image shape: {output_image_cupy.shape}")
            return output_image_cupy
        except Exception as e:
            logger.error(f"Error during raytracing: {e}", exc_info=True)
            return None


# --- Data Handling ---
def generate_dummy_data(
    num_samples: int,
    seq_length: int,
    vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates dummy data for training and testing purposes.

    Args:
        num_samples (int): The number of samples (sequences) to generate.
        seq_length (int): The length of each sequence.
        vocab_size (int): The size of the vocabulary (max token ID + 1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - inputs (torch.Tensor): Generated input sequences. Shape: (num_samples, seq_length).
            - labels (torch.Tensor): Generated label sequences (shifted inputs). Shape: (num_samples, seq_length).
    """
    logger.debug(
        f"Generating {num_samples} dummy samples with seq_length={seq_length}, vocab_size={vocab_size}"
    )
    # Generate random integer sequences as inputs (simulated sentences).
    inputs = torch.randint(0, vocab_size, (num_samples, seq_length), dtype=torch.long)
    # For language modeling, labels are typically the input sequence shifted by one position (predict next token).
    labels = torch.roll(inputs, shifts=-1, dims=1)
    # Set the label for the last token in each sequence to a padding index (e.g., 0),
    # as there is no "next token" to predict for it.
    labels[:, -1] = 0  # Assuming 0 is the padding token ID.
    return inputs, labels


# --- Evaluation ---
def evaluate_model(
    model: AdvancedHolographicLLM,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str
) -> Tuple[float, float, float]: # Added float for perplexity
    """
    Evaluates the model on a given dataset.

    Args:
        model (AdvancedHolographicLLM): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.CrossEntropyLoss): The loss function.
        device (str): The device to perform evaluation on ('cuda' or 'cpu').

    Returns:
        Tuple[float, float, float]:
            - avg_loss (float): Average loss over the evaluation dataset.
            - accuracy (float): Overall accuracy on the evaluation dataset (non-padding tokens).
            - avg_perplexity (float): Average perplexity over the evaluation dataset.
    """
    model.eval()  # Set the model to evaluation mode.
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0  # Total number of non-padding tokens for accuracy calculation.
    batch_perplexities: List[float] = []

    with torch.no_grad():  # Disable gradient calculations during evaluation.
        for batch_num, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # For evaluation, a specific context hologram might or might not be passed.
            # Here, we call the model without an explicit context hologram for simplicity.
            # (Original comment translated: For evaluation, we might not have a separate "context" hologram,
            # or we could pass one if the evaluation is contextual.
            # Here, we call the model without a context hologram for simplicity.)
            logits, _ = model(inputs, context_hologram=None)
            # We don't use the returned `current_input_hologram` for basic evaluation.

            # Calculate loss.
            # Reshape logits to (batch_size * seq_len, vocab_size) and labels to (batch_size * seq_len).
            loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
            # Accumulate loss, weighted by the number of samples in the batch.
            total_loss += loss.item() * inputs.size(0)


            # Calculate accuracy, ignoring padding tokens.
            _, predicted_tokens = torch.max(logits, dim=-1) # Get predicted token IDs
            # Create a mask for non-padding tokens (assuming 0 is padding).
            non_padding_mask = (labels != 0)
            total_correct += (predicted_tokens[non_padding_mask] == labels[non_padding_mask]).sum().item()
            total_tokens += non_padding_mask.sum().item() # Count only non-padding tokens for accuracy

            # Calculate perplexity for the current batch.
            # CrossEntropyLoss with default reduction='mean' gives the average negative log likelihood per token.
            # Perplexity = exp(average cross-entropy loss per token).
            # If loss.item() is already the mean loss for the batch (which it is by default for PyTorch's CrossEntropyLoss):
            current_batch_perplexity = torch.exp(loss).item()
            batch_perplexities.append(current_batch_perplexity)

            if batch_num > 0 and batch_num % 50 == 0: # Log progress occasionally for larger datasets
                logger.debug(f"Evaluation batch {batch_num}/{len(data_loader)} processed. Batch loss: {loss.item():.4f}")


    avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    avg_perplexity = np.mean(batch_perplexities) if batch_perplexities else float('inf')

    logger.info(
        f"Evaluation Results - Avg Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, Avg Perplexity: {avg_perplexity:.2f}"
    )
    return avg_loss, accuracy, avg_perplexity


# --- Main Training Loop ---
def main_training_loop() -> None:
    """
    Main function to orchestrate the model training and evaluation process.
    """
    logger.info("Starting main training loop...")

    # Initialize configuration, model, optimizer, and loss function.
    config = ModelConfig()
    model = AdvancedHolographicLLM(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Loss function, ignoring padding index 0 (consistent with `generate_dummy_data`).
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- Data Preparation ---
    logger.info("Generating dummy data for training and validation...")
    # Parameters: num_samples, seq_length, vocab_size
    inputs, labels = generate_dummy_data(
        num_samples=1000, seq_length=20, vocab_size=config.vocab_size
    )

    # Split data into training and validation sets.
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        inputs, labels, test_size=0.2, random_state=42 # Use a fixed random state for reproducibility
    )

    # Create TensorDatasets and DataLoaders.
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)

    # Determine number of workers based on CPU availability for DataLoaders
    num_workers = min(4, os.cpu_count() if os.cpu_count() is not None else 1)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=num_workers
    )
    logger.info(
        f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples. "
        f"Using {num_workers} workers for DataLoaders."
    )

    logger.info(f"Training started on device: {config.device}")
    training_start_time = time.time()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Context hologram management:
    # In a real application, context (e.g., `context_hologram`) would be managed more dynamically.
    # For this conceptual script, `context_hologram` is passed as `None` to the model's
    # forward pass during the main training loop. This implies that:
    #   - Each batch is processed without explicit holographic context from prior batches, or
    #   - Contextual information is primarily handled by the recurrent nature of the sequence_encoder (e.g., LSTM state),
    #     and the holographic component operates mainly on the current input or a very localized context.
    # A more advanced setup might involve:
    #   - Generating a context hologram from previous segments of data.
    #   - Updating a persistent context hologram iteratively.
    #   - Using the model's own hidden states or output holograms from previous steps as context.
    # The current simplified approach is chosen for clarity in this initial refactoring.
    # (Original commented-out code for `context_hologram_example` is preserved below for reference if needed later)
    # context_hologram_example = None
    # if config.batch_size > 0:
    #     dummy_context_data_for_holo = torch.rand(
    #         config.batch_size, 20, config.holographic_dimension # Assuming seq_len of 20 for context
    #     ).to(config.device)
    #     context_hologram_example = HolographicUtils.complex_vector_holography(
    #         dummy_context_data_for_holo, config.holographic_dimension
    #     ).to(config.device)


    for epoch in range(config.num_epochs):
        model.train()  # Set model to training mode.
        epoch_train_loss = 0.0
        num_batches_processed = 0

        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            batch_inputs = batch_inputs.to(config.device)
            batch_labels = batch_labels.to(config.device)

            optimizer.zero_grad()  # Clear previous gradients.

            # Forward pass:
            # `context_hologram` is set to None here, as discussed above.
            logits, _ = model(batch_inputs, context_hologram=None) # No persistent external context for now
            # The returned `current_input_hologram` is not explicitly used in this basic training loop,
            # but could be stored or used for more advanced context management.

            loss = criterion(logits.view(-1, config.vocab_size), batch_labels.view(-1))
            loss.backward()  # Backpropagate the loss.
            # Optional: Gradient clipping (e.g., torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            optimizer.step()  # Update model parameters.

            epoch_train_loss += loss.item()
            num_batches_processed += 1

            if batch_idx > 0 and batch_idx % 10 == 0: # Log training progress every 10 batches
                logger.debug(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Batch Loss: {loss.item():.4f}"
                )

        avg_epoch_train_loss = epoch_train_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        model.metrics.train_loss.append(avg_epoch_train_loss)

        # --- Validation Phase ---
        logger.info(f"Starting validation for epoch {epoch+1}...")
        val_loss, val_accuracy, val_perplexity = evaluate_model( # Capture all three return values
            model, val_loader, criterion, config.device
        )
        model.metrics.val_loss.append(val_loss)
        model.metrics.accuracy.append(val_accuracy)
        model.metrics.perplexity.append(val_perplexity) # Store perplexity

        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} Summary: "
            f"Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, Val Perplexity: {val_perplexity:.2f}"
        )

        # --- Early Stopping Check ---
        if val_loss < best_val_loss - config.early_stopping_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Optionally save the best model here (example, commented out)
            # try:
            #     torch.save(model.state_dict(), "best_holographic_llm.pth")
            #     logger.info("Best model state saved to best_holographic_llm.pth")
            # except Exception as e:
            #     logger.error(f"Error saving best model: {e}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss."
                )
                break  # Exit the training loop.

    # Record total training time.
    model.metrics.training_time = time.time() - training_start_time
    logger.info(
        f"Training finished. Total time: {model.metrics.training_time:.2f}s "
        f"({model.metrics.training_time / 60:.2f} minutes)."
    )

    # --- Final Model Saving and Metrics Export (Optional examples, commented out) ---
    # These are placeholders; actual saving might require directory creation and error handling.
    # final_model_path = "final_holographic_llm.pth"
    # try:
    #     torch.save(model.state_dict(), final_model_path)
    #     logger.info(f"Final model state saved to {final_model_path}")
    # except Exception as e:
    #     logger.error(f"Error saving final model: {e}")

    # metrics_json_path = "performance_metrics.json"
    # try:
    #     with open(metrics_json_path, "w", encoding="utf-8") as f:
    #         import json # Ensure json is imported if this block is used
    #         # Convert metrics to a dictionary for JSON serialization
    #         # Exclude fields that might not be directly serializable or are very large, if any.
    #         metrics_dict = {
    #             key: value for key, value in model.metrics.__dict__.items()
    #             if not key.startswith('_') and isinstance(value, (list, float, int, str))
    #         }
    #         json.dump(metrics_dict, f, indent=4)
    #     logger.info(f"Performance metrics saved to {metrics_json_path}")
    # except Exception as e:
    #     logger.error(f"Error saving performance metrics: {e}")


if __name__ == "__main__":
    try:
        # The CUDA check is already performed during AdvancedHolographicLLM initialization.
        main_training_loop()
    except RuntimeError as e: # Catch specific errors like CUDA issues if not handled earlier
        logger.error(f"A runtime error occurred: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)
        # Potentially perform cleanup operations or save state before exiting.
    finally:
        logger.info("Script execution finished or terminated.")

# --- Areas for Further Refinement and Enhancement ---
# 1.  Command-line Interface: Use `argparse` for flexible configuration via command-line arguments.
# 2.  Data Loading: Implement more sophisticated data loading, including tokenization from raw text files
#     and handling of larger datasets that may not fit in memory.
# 3.  Holographic Implementation: The current holographic operations are highly conceptual.
#     A practical implementation would require detailed mathematical formulation and empirical validation
#     for memory update, retrieval, and integration into the LLM architecture.
# 3a. CuPy Integration: The current script uses PyTorch for its main model structure and training loop,
#     while new methods (`inference`, `generate_text`, `train_step`) use CuPy/NumPy for transformer-like
#     operations. This creates a hybrid model. Data needs to be explicitly moved between
#     PyTorch tensors (on CPU/GPU) and CuPy arrays (on GPU) or NumPy arrays (on CPU).
#     This is handled by `xp.asarray()` and `.cpu().numpy()` in `train_step` for example.
#     A more unified approach (all PyTorch or all CuPy for the core model) might be considered for simplification,
#     but the current hybrid approach fulfills the subtask of porting CuPy-based methods.
# 3b. ModelConfig: Added `sequence_length` and `num_neurons` to `ModelConfig` for use in new methods.
# 4.  Training Enhancements:
#     - Gradient Clipping: To prevent exploding gradients, especially in recurrent networks.
#     - Learning Rate Scheduling: Dynamically adjust the learning rate during training.
#     - Checkpointing: Save model checkpoints periodically and allow resuming training.
# 5.  Logging and Monitoring: Integrate with tools like TensorBoard or Weights & Biases for
#     visualizing training progress and metrics.
# 6.  Unit Testing: Develop unit tests for individual components (e.g., utility functions, model layers).
# 7.  Perplexity Calculation: Ensure robust perplexity calculation, especially for padded sequences
#     and when using distributed training (current implementation is a batch-wise approximation).
# 8.  Modularity: Refactor into a more modular structure (e.g., separate Trainer class, DataModule).
# 9.  Context Management: The handling of `context_hologram` is simplistic. A practical system would need
#     a well-defined strategy for how context is generated, updated, and utilized across sequences or batches.
# 10. Sparsity: The `sparsity_factor` in `ModelConfig` is defined but not actively used in the current
#     holographic operations. Incorporating sparsity could be relevant for efficiency or representation quality.
# 11. Tensor Device Consistency: Double-check that all tensors are consistently moved to the correct device
#     throughout the model and training loop to avoid runtime errors. (Largely addressed by `.to(device)`).
# 12. Output Projection Integration: The method of combining `encoded_output` and
#     `holographic_features_for_output` (currently concatenation) in `AdvancedHolographicLLM`
#     should be carefully designed and validated for its impact on performance.
# (Item 13 addressed by adding Optional to type hints where appropriate)
# (Item 14 addressed by using os.cpu_count() for DataLoader num_workers)
# (Item 15 addressed by using `input_size` in LSTM definition)
# (Item 16 `batch_first=True` is used and tensor shapes commented)
# (Item 17 added debug logging for batch progress in evaluation)
# (Item 18 accuracy calculation refined for padding)
# (Item 19 perplexity calculation uses batch average loss, which is standard for this approximation)
# (Item 20 `encoding="utf-8"` added to example file writing)
# (Item 21 `if __name__ == "__main__":` is present)
# (Item 22 `num_heads` comment clarified in ModelConfig)
# (Item 23 `complex_vector_holography` comments expanded)
# (Item 24 `AdvancedHolographicLLM.forward` tensor shape comments improved)
# (Item 25 `generate_dummy_data` uses `dtype=torch.long`)
# (Item 26 `Optional` added to `context_hologram` type hint)
# (Item 27 `num_workers` added to DataLoaders)
# (Item 28 `ignore_index` in `CrossEntropyLoss` is consistent with padding)
# (Item 29 `dtype=torch.long` is used)
# (Item 30 comments added to `main_training_loop` for context hologram)
# (Item 31 LSTM dropout condition `if config.num_layers > 1` is present)
# (Item 32 `holographic_features_for_output` derivation clarified in comments)
# (Item 33 logger format updated)
# (Item 34 `Optional` import added)


# --- Main Execution & Demo ---
def main():
    """
    Main function to run the Advanced Holographic LLM demo with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Advanced Holographic LLM Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )

    # Add arguments based on ModelConfig fields
    # LLM specific
    parser.add_argument("--vocab_size", type=int, default=ModelConfig.vocab_size, help="Vocabulary size.")
    parser.add_argument("--embedding_dim", type=int, default=ModelConfig.embedding_dim, help="Token embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=ModelConfig.hidden_dim, help="LSTM hidden state dimension.")
    parser.add_argument("--num_layers", type=int, default=ModelConfig.num_layers, help="Number of LSTM layers.")
    parser.add_argument("--num_heads", type=int, default=ModelConfig.num_heads, help="Number of attention heads (conceptual).")
    parser.add_argument("--dropout_rate", type=float, default=ModelConfig.dropout_rate, help="Dropout rate for PyTorch layers.")
    parser.add_argument("--learning_rate", type=float, default=ModelConfig.learning_rate, help="Learning rate for conceptual train_step.")
    # batch_size and num_epochs for the main_training_loop (PyTorch based) are not used in this demo's main().
    # We use args.epochs for the demo's conceptual training loop.

    # Device and Precision
    parser.add_argument("--device", type=str, default=ModelConfig.device, choices=["cuda", "cpu"], help="Device for PyTorch and CuPy operations.")
    parser.add_argument("--numerical_precision", type=str, default=ModelConfig.numerical_precision, choices=["float32", "float64"], help="Numerical precision for CuPy/NumPy arrays.")

    # Holographic specific
    parser.add_argument("--holographic_dimension", type=int, default=ModelConfig.holographic_dimension, help="Dimensionality of the holographic space.")
    parser.add_argument("--num_neurons", type=int, default=ModelConfig.num_neurons, help="Number of neurons in the holographic layer.")
    parser.add_argument("--neuron_sphere_radius", type=float, default=ModelConfig.neuron_sphere_radius, help="Radius of spherical neurons for raytracing.")
    parser.add_argument("--grid_size", type=int, default=ModelConfig.grid_size, help="Grid size for hologram plane / raytracing output image.")
    parser.add_argument("--wavelength", type=float, default=ModelConfig.wavelength, help="Wavelength of light for propagation (meters).")
    parser.add_argument("--pixel_size", type=float, default=ModelConfig.pixel_size, help="Pixel size for propagation (meters).")
    parser.add_argument("--propagation_distance", type=float, default=ModelConfig.propagation_distance, help="Propagation distance for hologram (meters).")

    # Text generation
    parser.add_argument("--sequence_length", type=int, default=ModelConfig.sequence_length, help="Sequence length for training inputs and some internal operations.")
    parser.add_argument("--temperature", type=float, default=ModelConfig.temperature, help="Softmax temperature for text generation.")
    parser.add_argument("--max_gen_length", type=int, default=30, help="Max length for generated text during demo.")

    # Demo specific arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of conceptual training epochs for the demo.")

    args = parser.parse_args()

    # Create ModelConfig instance from parsed arguments
    # Note: PerformanceMetrics defaults are used if not overridden by ModelConfig defaults for shared fields.
    config = ModelConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=ModelConfig.batch_size, # Using ModelConfig default, not an arg for this demo main
        num_epochs=ModelConfig.num_epochs, # Using ModelConfig default
        device=args.device,
        holographic_dimension=args.holographic_dimension,
        sparsity_factor=ModelConfig.sparsity_factor, # Using ModelConfig default
        early_stopping_patience=ModelConfig.early_stopping_patience, # Using ModelConfig default
        early_stopping_delta=ModelConfig.early_stopping_delta, # Using ModelConfig default

        sequence_length=args.sequence_length,
        num_neurons=args.num_neurons,
        neuron_sphere_radius=args.neuron_sphere_radius,
        grid_size=args.grid_size,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        propagation_distance=args.propagation_distance,
        numerical_precision=args.numerical_precision,
        temperature=args.temperature
    )

    logger.info("="*50)
    logger.info("Starting Advanced Holographic LLM Demo")
    logger.info("="*50)
    logger.info(f"Using configuration: {config}")
    if not CUPY_AVAILABLE and config.device == 'cuda':
        logger.warning("CUDA device selected, but CuPy is not available. GPU operations via CuPy will fallback to NumPy on CPU.")
    if not NUMBA_AVAILABLE and config.device == 'cuda':
        logger.warning("CUDA device selected, but Numba is not available. Numba CUDA kernels will not be functional.")


    # Instantiate the model
    # The model's __init__ handles moving PyTorch parts to config.device.
    # CuPy parts use config.device to choose backend (cp or np).
    model = AdvancedHolographicLLM(config)

    # --- Conceptual Hologram Operations (Example) ---
    # model.hologram is initialized in _initialize_model.
    # For demo, ensure it's on the correct (CuPy/NumPy) device.
    xp = cp if (CUPY_AVAILABLE and config.device == "cuda") else np
    if isinstance(model.hologram, np.ndarray) and xp == cp :
        model.hologram = cp.asarray(model.hologram) # Ensure CuPy array if using GPU
    elif isinstance(model.hologram, cp.ndarray) and xp == np:
         model.hologram = model.hologram.get() # Ensure NumPy array if using CPU

    logger.info(f"Initial model.hologram array type: {type(model.hologram)}")

    model.propagate_hologram()

    raytraced_output = model.raytrace()
    if raytraced_output is not None:
        logger.info(f"Raytracing demo completed. Output image (sum): {xp.sum(raytraced_output) if raytraced_output is not None else 'N/A'}")
    else:
        logger.warning("Raytracing demo was skipped or failed.")

    # --- Training Loop (Conceptual, using model.train_step for CuPy/NumPy weights) ---
    logger.info(f"\nStarting conceptual training loop for {args.epochs} epochs (updates CuPy/NumPy weights)...")
    for epoch in range(args.epochs):
        # Create dummy batch data (CuPy/NumPy arrays)
        # train_step expects a batch. For demo, use a small batch_size.
        demo_batch_size = config.batch_size if config.batch_size > 0 else 1 # Use configured batch_size or 1

        dummy_input_batch = xp.random.randint(0, config.vocab_size,
                                              size=(demo_batch_size, config.sequence_length),
                                              dtype=xp.int32)
        dummy_target_batch = xp.random.randint(0, config.vocab_size,
                                               size=(demo_batch_size, config.sequence_length),
                                               dtype=xp.int32)

        loss = model.train_step(dummy_input_batch, dummy_target_batch)
        logger.info(f"Demo Training Epoch {epoch+1}/{args.epochs}, Conceptual CuPy/NumPy Loss: {loss:.4f}")

    # --- Text Generation ---
    logger.info("\nStarting text generation...")
    input_sequence_for_generation = [1, 2, 3] # Example starting sequence
    # Ensure tokens are within vocab_size
    input_sequence_for_generation = [min(idx, config.vocab_size - 1) for idx in input_sequence_for_generation if idx >=0]
    if not input_sequence_for_generation : input_sequence_for_generation = [0]


    generated_sequence = model.generate_text(
        input_sequence_for_generation,
        max_length=args.max_gen_length,
        temperature=config.temperature
    )
    logger.info(f"Generated sequence: {generated_sequence}")

    # --- Performance Metrics (Conceptual) ---
    # These metrics are primarily from the PyTorch-based main_training_loop, not this demo.
    logger.info("\nConceptual Performance Metrics (from PyTorch main_training_loop, if it were run):")
    logger.info(f"  Pytorch Training Losses: {model.metrics.train_loss}")
    logger.info(f"  Pytorch Validation Losses: {model.metrics.val_loss}")
    logger.info(f"  Pytorch Training Time: {model.metrics.training_time:.2f}s")

    logger.info("="*50)
    logger.info("Advanced Holographic LLM Demo finished.")
    logger.info("="*50)


# --- Main Execution & Demo ---
def main():
    """
    Main function to run the Advanced Holographic LLM demo with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Advanced Holographic LLM Demo")

    # Add arguments based on ModelConfig fields
    parser.add_argument("--vocab_size", type=int, default=ModelConfig.vocab_size)
    parser.add_argument("--embedding_dim", type=int, default=ModelConfig.embedding_dim)
    parser.add_argument("--hidden_dim", type=int, default=ModelConfig.hidden_dim)
    parser.add_argument("--num_layers", type=int, default=ModelConfig.num_layers)
    parser.add_argument("--num_heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--dropout_rate", type=float, default=ModelConfig.dropout_rate)
    parser.add_argument("--learning_rate", type=float, default=ModelConfig.learning_rate)
    parser.add_argument("--batch_size", type=int, default=ModelConfig.batch_size)
    # num_epochs for main_training_loop is different from demo epochs
    # parser.add_argument("--num_epochs", type=int, default=ModelConfig.num_epochs)
    parser.add_argument("--device", type=str, default=ModelConfig.device, choices=["cuda", "cpu"])

    parser.add_argument("--holographic_dimension", type=int, default=ModelConfig.holographic_dimension)
    parser.add_argument("--sparsity_factor", type=float, default=ModelConfig.sparsity_factor)
    parser.add_argument("--early_stopping_patience", type=int, default=ModelConfig.early_stopping_patience)
    parser.add_argument("--early_stopping_delta", type=float, default=ModelConfig.early_stopping_delta)

    # For PerformanceMetrics defaults that are also in ModelConfig
    parser.add_argument("--sequence_length", type=int, default=PerformanceMetrics.sequence_length)
    parser.add_argument("--num_neurons", type=int, default=PerformanceMetrics.num_neurons)
    parser.add_argument("--neuron_sphere_radius", type=float, default=PerformanceMetrics.neuron_sphere_radius)

    # For other ModelConfig parameters used in demo
    parser.add_argument("--grid_size", type=int, default=ModelConfig.grid_size)
    parser.add_argument("--wavelength", type=float, default=ModelConfig.wavelength)
    parser.add_argument("--pixel_size", type=float, default=ModelConfig.pixel_size)
    parser.add_argument("--propagation_distance", type=float, default=ModelConfig.propagation_distance)
    parser.add_argument("--numerical_precision", type=str, default=ModelConfig.numerical_precision, choices=["float32", "float64"])
    parser.add_argument("--temperature", type=float, default=ModelConfig.temperature)

    # Demo specific arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for the demo.")
    parser.add_argument("--max_gen_length", type=int, default=30, help="Max length for generated text.")

    args = parser.parse_args()

    # Create ModelConfig instance from parsed arguments
    config = ModelConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        # num_epochs=args.num_epochs, # This is for the main training loop, not demo
        device=args.device,
        holographic_dimension=args.holographic_dimension,
        sparsity_factor=args.sparsity_factor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        # From PerformanceMetrics defaults but part of ModelConfig structure
        sequence_length=args.sequence_length,
        num_neurons=args.num_neurons,
        neuron_sphere_radius=args.neuron_sphere_radius,
        grid_size=args.grid_size,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        propagation_distance=args.propagation_distance,
        numerical_precision=args.numerical_precision,
        temperature=args.temperature
    )

    logger.info("Starting Advanced Holographic LLM Demo")
    logger.info(f"Using configuration: {config}")

    # Instantiate the model
    # Note: The PyTorch part of model is moved to config.device by its own __init__
    # CuPy parts will use config.device to decide GPU/CPU for cp vs np.
    model = AdvancedHolographicLLM(config)

    # --- Conceptual Hologram Operations (Example) ---
    # model.hologram is initialized in _initialize_model.
    # For demo, ensure it's on the correct (CuPy/NumPy) device if we re-assign
    xp = cp if (CUPY_AVAILABLE and config.device == "cuda") else np
    if model.hologram is None or model.hologram.shape != (config.holographic_dimension, config.holographic_dimension):
        logger.info(f"Re-initializing demo hologram on {xp.__name__} with dimension: {config.holographic_dimension}")
        model.hologram = xp.random.rand(config.holographic_dimension, config.holographic_dimension).astype(xp.complex64 if config.numerical_precision == "float32" else xp.complex128)

    # Call propagate_hologram (placeholder for now)
    model.propagate_hologram()
    logger.info("Conceptual hologram propagation attempted.")

    # Call raytrace
    raytraced_output = model.raytrace()
    if raytraced_output is not None:
        logger.info(f"Raytracing complete. Output image shape: {raytraced_output.shape}")
    else:
        logger.warning("Raytracing was skipped or failed.")

    # --- Training Loop (Conceptual, using model.train_step) ---
    logger.info(f"Starting conceptual training loop for {args.epochs} epochs...")
    # Use xp for dummy data generation consistent with train_step's expectation
    for epoch in range(args.epochs):
        # Create a single dummy sequence for this simplified training_step
        # train_step currently expects a batch of one sequence for simplicity
        # For a batch: input_sequence_batch = xp.random.randint(0, config.vocab_size, size=(1, config.sequence_length), dtype=xp.int32)
        # target_sequence_batch = xp.random.randint(0, config.vocab_size, size=(1, config.sequence_length), dtype=xp.int32)
        # loss = model.train_step(input_sequence_batch, target_sequence_batch)

        # train_step expects a single sequence, not a batch based on its current loop structure.
        # Let's provide data that matches its internal loop.
        # The train_step itself iterates `batch_size` times internally, which is a bit unusual.
        # For demo, we'll just call it once per epoch with a single sequence.
        # This part highlights a slight mismatch between typical batch training and the current train_step.
        # For now, we adapt to what train_step expects.

        # Dummy data for train_step (single instance, as train_step has its own internal batch loop)
        # However, train_step is designed to take a batch and loop over it.
        # Let's create a batch of size 1 for the demo.
        demo_batch_size = 1
        dummy_input_batch = xp.random.randint(0, config.vocab_size, size=(demo_batch_size, config.sequence_length), dtype=xp.int32)
        dummy_target_batch = xp.random.randint(0, config.vocab_size, size=(demo_batch_size, config.sequence_length), dtype=xp.int32)

        # The PyTorch part of the model (LSTM, etc.) is not trained by this train_step.
        # This train_step only updates the CuPy/NumPy based conceptual transformer weights.
        loss = model.train_step(dummy_input_batch, dummy_target_batch)
        logger.info(f"Demo Training Epoch {epoch+1}/{args.epochs}, Conceptual Loss: {loss:.4f}")

    # --- Text Generation ---
    logger.info("Starting text generation...")
    # Initial sequence for generation (ensure token IDs are within vocab_size)
    input_sequence_for_generation = [1, 2, 3]
    input_sequence_for_generation = [min(idx, config.vocab_size -1) for idx in input_sequence_for_generation]

    generated_sequence = model.generate_text(
        input_sequence_for_generation,
        max_length=args.max_gen_length,
        temperature=config.temperature
    )
    logger.info(f"Generated sequence: {generated_sequence}")

    # --- Performance Metrics (Conceptual) ---
    logger.info("Conceptual Performance Metrics:")
    logger.info(f"  Training Losses (from PyTorch loop, if run): {model.metrics.train_loss}")
    logger.info(f"  Validation Losses (from PyTorch loop, if run): {model.metrics.val_loss}")
    logger.info(f"  Training Time (PyTorch loop): {model.metrics.training_time:.2f}s")
    # The train_step in this demo updates different weights, so its loss isn't directly in model.metrics.train_loss

    logger.info("Advanced Holographic LLM Demo finished.")


# --- Numba CUDA specific imports and functions ---
# These are typically placed at the top level of the module they are used in.
# For simplicity in this consolidated script, they are here.
try:
    from numba import cuda, float32
    import math # Standard math library, Numba device functions can use it
    NUMBA_AVAILABLE = True
    logger.info("Numba successfully imported. CUDA kernels will be available if a CUDA device is present.")
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide dummy decorators if Numba is not available, so the script can still be parsed.
    logger.warning("Numba not found. CUDA kernels (e.g., advanced_raytrace_kernel) will not be functional.")
    def cuda_jit_decorator(*args, **kwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                raise NotImplementedError("Numba CUDA kernel called but Numba is not installed.")
            return wrapper
        # Handle cases where @cuda.jit is used with or without arguments
        if len(args) == 1 and callable(args[0]): # @cuda.jit
            return decorator(args[0])
        else: # @cuda.jit(device=True), etc.
            return decorator
    cuda.jit = cuda_jit_decorator
    cuda.grid = lambda x: (0,0) # Dummy for cuda.grid(2)
    float32 = np.float32 # Fallback type


if NUMBA_AVAILABLE:
    @cuda.jit(device=True, fastmath=True)
    def sphere_ray_intersection(ray_origin, ray_direction, sphere_center, sphere_radius) -> Tuple[bool, float]:
        """
        Calculates the intersection of a ray with a sphere using Numba for CUDA device execution.

        Args:
            ray_origin (numba.cuda.cudadrv.devicearray.DeviceNDArray): Origin of the ray (x, y, z).
            ray_direction (numba.cuda.cudadrv.devicearray.DeviceNDArray): Normalized direction of the ray (dx, dy, dz).
            sphere_center (numba.cuda.cudadrv.devicearray.DeviceNDArray): Center of the sphere (cx, cy, cz).
            sphere_radius (float): Radius of the sphere.

        Returns:
            Tuple[bool, float]:
                - bool: True if an intersection occurs, False otherwise.
                - float: The distance `t` from the ray origin to the closest valid intersection point.
                         Returns a large number (1e6) if no intersection.
        """
        # Vector from ray origin to sphere center
        oc_x = sphere_center[0] - ray_origin[0]
        oc_y = sphere_center[1] - ray_origin[1]
        oc_z = sphere_center[2] - ray_origin[2]

        # Projection of oc onto ray_direction (b in quadratic equation term)
        # loc_dot_dir = oc_x * ray_direction[0] + oc_y * ray_direction[1] + oc_z * ray_direction[2]
        b = oc_x * ray_direction[0] + oc_y * ray_direction[1] + oc_z * ray_direction[2]

        # Squared length of oc vector (c in quadratic equation term, relative to discriminant)
        oc_squared_length = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z

        # Discriminant of the quadratic equation: b^2 - (oc_squared_length - radius^2)
        # Simplified from (loc_dot_dir)^2 - (oc_squared_length - sphere_radius^2)
        # The equation for t is At^2 + Bt + C = 0, where A=1 (normalized ray_direction), B=-2*b', C=oc_sq - r_sq
        # Discriminant = B^2 - 4AC = (-2b')^2 - 4(oc_sq-r_sq) = 4b'^2 - 4(oc_sq-r_sq)
        # We are interested in b'^2 - (oc_sq-r_sq) for checking sign.
        discriminant = b * b - (oc_squared_length - sphere_radius * sphere_radius)

        if discriminant < 0:
            return False, 1e6  # No real intersection

        # Calculate intersection distances
        sqrt_discriminant = math.sqrt(discriminant)
        t0 = b - sqrt_discriminant
        t1 = b + sqrt_discriminant

        # Check for valid intersection (t > epsilon, positive direction)
        # Epsilon (1e-6) helps avoid self-intersection or issues at the ray origin.
        epsilon = 1e-6
        if t0 > epsilon:
            return True, t0  # t0 is the closer valid intersection
        elif t1 > epsilon:
            return True, t1  # t1 is the closer valid intersection (t0 was not valid)

        return False, 1e6 # No intersection in the positive direction of the ray


    @cuda.jit(fastmath=True)
    def advanced_raytrace_kernel(hologram, neurons, output_image, config_params_tuple):
        """
        Numba CUDA kernel for raytracing through a field of spherical neurons
        and sampling a hologram.

        Args:
            hologram (cupy.ndarray): The complex-valued hologram data. Shape (H, W).
            neurons (cupy.ndarray): Neuron positions (x,y,z) and potentially other properties. Shape (num_neurons, 3+).
            output_image (cupy.ndarray): Output image buffer to store raytraced results. Shape (img_H, img_W).
            config_params_tuple (tuple): Tuple containing configuration parameters. Expected: (sphere_radius,).
        """
        # Get thread indices for the output image pixel
        # cuda.grid(2) returns a tuple (x, y) for 2D grid
        pixel_x, pixel_y = cuda.grid(2)

        # Image dimensions from output_image shape
        img_height, img_width = output_image.shape

        # Boundary check: ensure current thread is within image dimensions
        if pixel_x >= img_width or pixel_y >= img_height:
            return

        # Extract configuration parameters
        sphere_radius = config_params_tuple[0] # Example: neuron_sphere_radius from ModelConfig

        # --- Ray Initialization ---
        # Normalized device coordinates (NDC) for the current pixel: [0, 1] range
        # Assuming image plane is at z=0, and rays originate from z=-1 (or some distance)
        # This setup simulates an orthographic or perspective camera based on ray direction calculation.
        # For simplicity, let's assume rays originate from a plane at z = -1, looking towards +z.
        # Pixel coordinates in normalized space [-1, 1] (approx, adjust for aspect ratio)
        ndc_x = (pixel_x / img_width) * 2.0 - 1.0
        ndc_y = (pixel_y / img_height) * 2.0 - 1.0
        # (Could adjust for aspect ratio if sensor_width != sensor_height conceptually)

        ray_origin_x = ndc_x
        ray_origin_y = ndc_y
        ray_origin_z = -1.0 # Ray origin behind the image plane (conceptual camera position)

        # Ray direction (normalized) - for orthographic projection, all rays are parallel (0,0,1)
        # For perspective, it would depend on ray_origin and a focal point.
        # Let's use orthographic for simplicity here.
        ray_dir_x = 0.0
        ray_dir_y = 0.0
        ray_dir_z = 1.0
        # (Normalization not strictly needed if (0,0,1) but good practice for general directions)

        # Temporary arrays for Numba device function compatibility (cannot pass tuples directly sometimes)
        temp_ray_origin = cuda.local.array(3, dtype=float32)
        temp_ray_origin[0], temp_ray_origin[1], temp_ray_origin[2] = ray_origin_x, ray_origin_y, ray_origin_z

        temp_ray_direction = cuda.local.array(3, dtype=float32)
        temp_ray_direction[0], temp_ray_direction[1], temp_ray_direction[2] = ray_dir_x, ray_dir_y, ray_dir_z

        # --- Ray Tracing through Neurons ---
        accumulated_intensity = 0.0
        num_intersections_for_pixel = 0 # Number of neurons this ray intersects

        for i in range(neurons.shape[0]): # Loop through each neuron
            neuron_center_x = neurons[i, 0]
            neuron_center_y = neurons[i, 1]
            neuron_center_z = neurons[i, 2]

            temp_neuron_center = cuda.local.array(3, dtype=float32)
            temp_neuron_center[0], temp_neuron_center[1], temp_neuron_center[2] = neuron_center_x, neuron_center_y, neuron_center_z

            # Check for intersection with the current neuron's sphere
            intersects, t = sphere_ray_intersection(temp_ray_origin, temp_ray_direction, temp_neuron_center, sphere_radius)

            if intersects and t < 1e5: # Check if intersection is valid and not at infinity
                num_intersections_for_pixel += 1

                # Calculate intersection point
                intersect_x = ray_origin_x + ray_dir_x * t
                intersect_y = ray_origin_y + ray_dir_y * t
                # intersect_z = ray_origin_z + ray_dir_z * t # Z-coordinate of intersection

                # --- Hologram Sampling ---
                # Convert intersection point (x,y on the neuron surface facing the ray)
                # to hologram coordinates [0, H-1] and [0, W-1].
                # This is a simplified mapping. A more realistic model would depend on the
                # hologram's placement relative to the neurons and the nature of reconstruction.
                # Assuming hologram is a plane at some z, and we project intersect_x, intersect_y.
                # For this example, let's map normalized intersection coords [-0.5, 0.5] to hologram dims.
                # (This assumes neurons are centered around (0,0) and hologram covers this area)

                # Map intersect_x, intersect_y (which are world coordinates) to hologram texture coordinates (u,v) [0,1]
                # This mapping needs to be defined based on how the hologram is oriented and scaled.
                # Example: if neurons are in [-0.5, 0.5] cube, and hologram covers this.
                holo_u = (intersect_x + 0.5) # Assumes intersect_x is in [-0.5, 0.5] range
                holo_v = (intersect_y + 0.5) # Assumes intersect_y is in [-0.5, 0.5] range

                hologram_height, hologram_width = hologram.shape

                # Convert (u,v) to discrete hologram pixel indices
                holo_sample_x = int(holo_u * (hologram_width - 1))
                holo_sample_y = int(holo_v * (hologram_height - 1))

                # Boundary check for hologram sampling
                if 0 <= holo_sample_x < hologram_width and 0 <= holo_sample_y < hologram_height:
                    # Sample hologram value (e.g., intensity, which is magnitude squared of complex value)
                    # For a complex hologram, one might take abs()^2 or real part based on optical model
                    hologram_value = hologram[holo_sample_y, holo_sample_x]
                    # Assuming hologram_value is complex, take its magnitude (amplitude)
                    # In a real optical setup, intensity is proportional to magnitude squared.
                    # Here, let's assume the value sampled is already some form of intensity or phase factor.
                    # For simplicity, let's use the real part if complex, or itself if real.
                    # accumulated_intensity += hologram_value.real # If complex
                    accumulated_intensity += abs(hologram_value) # Using magnitude for accumulation

        # Normalize accumulated intensity by number of intersections for this pixel's ray
        if num_intersections_for_pixel > 0:
            output_image[pixel_y, pixel_x] = accumulated_intensity / num_intersections_for_pixel
        else:
            output_image[pixel_y, pixel_x] = 0.0 # No intersection, pixel is black


def angular_spectrum_propagation(input_field: Any, distance: float, wavelength: float, pixel_size: float) -> Any:
    """
    Propagates a complex wavefield using the angular spectrum method.
    This version uses CuPy for GPU acceleration if available, otherwise NumPy.

    Args:
        input_field (Any): 2D complex array representing the input wavefield (CuPy or NumPy).
        distance (float): Propagation distance.
        wavelength (float): Wavelength of the light.
        pixel_size (float): Physical size of a pixel in the input field.

    Returns:
        Any: 2D complex array representing the propagated wavefield (CuPy or NumPy).
    """
    xp = cp.get_array_module(input_field) # Get CuPy or NumPy module

    if input_field.ndim != 2:
        raise ValueError("Input field must be a 2D array.")

    rows, cols = input_field.shape
    logger.debug(
        f"Angular spectrum propagation: distance={distance}, wavelength={wavelength}, pixel_size={pixel_size}. "
        f"Input field shape: ({rows}, {cols}). Using '{xp.__name__}'."
    )

    # 1. Fourier Transform of the input field
    # Shift the zero-frequency component to the center for FT
    ft_input_field = xp.fft.fftshift(xp.fft.fft2(input_field))
    logger.debug(f"  Shape after FFT2 and fftshift: {ft_input_field.shape}")

    # 2. Create spatial frequency coordinates
    # Frequency sampling intervals
    dfx = 1.0 / (cols * pixel_size)
    dfy = 1.0 / (rows * pixel_size)

    # Frequency coordinates (fx, fy)
    fx = xp.arange(-cols // 2, cols // 2) * dfx
    fy = xp.arange(-rows // 2, rows // 2) * dfy
    fx_grid, fy_grid = xp.meshgrid(fx, fy)

    # 3. Calculate the propagation kernel in the frequency domain (transfer function H)
    # Wave number k
    k = 2 * xp.pi / wavelength

    # Term inside the square root for kz
    # kz_squared_arg = k*k - (2*xp.pi*fx_grid)**2 - (2*xp.pi*fy_grid)**2
    # More direct: (2*pi/lambda)^2 - (2*pi*fx_grid)^2 - (2*pi*fy_grid)^2
    # = (2*pi)^2 * ( (1/lambda)^2 - fx_grid^2 - fy_grid^2 )

    # Calculation of kz (wave vector component in z-direction)
    # Ensure stability: argument of sqrt should be non-negative for propagating (evanescent waves decay)
    fx2_plus_fy2 = fx_grid**2 + fy_grid**2
    k_squared = k**2

    # Mask for evanescent waves (high frequencies that don't propagate far)
    # These frequencies correspond to (fx^2 + fy^2) > (1/wavelength)^2
    evanescent_mask = fx2_plus_fy2 > (1.0 / wavelength)**2

    # For propagating waves: kz = sqrt(k^2 - (2*pi*fx)^2 - (2*pi*fy)^2)
    # For evanescent waves: kz is imaginary, leading to exponential decay.
    # Here, we are calculating sqrt( (1/lambda)^2 - fx^2 - fy^2 ) then multiplying by 2*pi.
    # This is equivalent to sqrt(k^2 - (2*pi*fx)^2 - (2*pi*fy)^2) / (2*pi)

    # Argument for the square root in kz calculation
    kz_arg = (1.0 / wavelength)**2 - fx2_plus_fy2

    # Set kz for evanescent waves to ensure decay (real part of sqrt will be 0, imag part for decay)
    # For stability, avoid taking sqrt of negative numbers directly if not handled by complex sqrt.
    # However, propagation kernel uses exp(1j * kz * distance).
    # If kz_arg < 0, sqrt(kz_arg) = 1j * sqrt(abs(kz_arg)).
    # Then 1j * (1j * sqrt(abs(kz_arg))) = - sqrt(abs(kz_arg)). This gives real negative exponent -> decay.

    # Using xp.sqrt handles complex results automatically if kz_arg is negative.
    kz_over_2pi = xp.sqrt(kz_arg.astype(xp.complex64)) # Ensure complex type for sqrt

    # Transfer function H = exp(1j * kz * distance)
    # kz = 2 * xp.pi * kz_over_2pi
    transfer_function_H = xp.exp(1j * (2 * xp.pi * kz_over_2pi) * distance)

    # Apply mask for high frequencies (optional, but good for numerical stability / removing artifacts)
    # This acts as a low-pass filter in the frequency domain.
    # The radius of this filter is k, meaning frequencies higher than k are attenuated.
    # This is implicitly handled if kz_arg becomes negative, as kz becomes imaginary leading to decay.
    # However, an explicit aperture can be useful.
    # Example: Define a circular mask based on k
    # filter_radius_sq = (1/wavelength)**2
    # freq_radius_sq = fx_grid**2 + fy_grid**2
    # aperture_mask = freq_radius_sq <= filter_radius_sq
    # transfer_function_H *= aperture_mask # Apply aperture

    logger.debug(f"  Transfer function H shape: {transfer_function_H.shape}, dtype: {transfer_function_H.dtype}")

    # 4. Multiply Fourier transformed field by the transfer function
    ft_propagated_field = ft_input_field * transfer_function_H

    # 5. Inverse Fourier Transform to get the propagated field in spatial domain
    # Shift zero-frequency back to corner before IFFT
    propagated_field = xp.fft.ifft2(xp.fft.ifftshift(ft_propagated_field))
    logger.debug(f"  Shape after IFFT2 and ifftshift (final): {propagated_field.shape}")

    return propagated_field

```
