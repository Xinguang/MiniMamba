import torch
import time
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from minimamba import Mamba
from minimamba import MambaConfig

def main():
    # ----------------------------------------
    # 1. Define model configuration
    # ----------------------------------------
    config = MambaConfig(
        d_model=512,
        n_layer=6,
        vocab_size=10000,
        d_state=16,
        d_conv=4,
        expand=2,
    )

    # ----------------------------------------
    # 2. Select device (MPS > CUDA > CPU)
    # ----------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using device: CPU")

    # ----------------------------------------
    # 3. Initialize model and inputs
    # ----------------------------------------
    model = Mamba(config=config).to(device)

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")

    # ----------------------------------------
    # 4. Run forward pass and measure time
    # ----------------------------------------
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids)
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    print(f"Output shape: {logits.shape}")
    print(f"Inference time: {(end_time - start_time):.4f} seconds")

if __name__ == "__main__":
    main()