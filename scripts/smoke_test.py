"""Verify the environment is healthy: Python, PyTorch, MPS backend."""
import sys


def main() -> int:
    print(f"Python:     {sys.version.split()[0]}")

    try:
        import torch
    except ImportError as e:
        print(f"FAIL: could not import torch ({e})")
        return 1

    print(f"PyTorch:    {torch.__version__}")
    print(f"MPS built:  {torch.backends.mps.is_built()}")
    print(f"MPS avail:  {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        print("WARN: MPS not available. Course material assumes Apple Silicon GPU.")
        return 0

    device = torch.device("mps")
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)
    c = a @ b
    assert c.shape == (64, 64)
    print(f"MPS matmul: OK (output shape {tuple(c.shape)})")

    import g2c
    print(f"g2c:        {g2c.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
