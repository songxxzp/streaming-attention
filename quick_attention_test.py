#!/usr/bin/env python3
"""
Quick Attention Validation Test

Uses PyTorch's F.scaled_dot_product_attention to validate
self-attention and cross-attention implementations.
"""

import torch
import numpy as np
import sys

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def test_self_attention():
    """Test self-attention with PyTorch reference"""
    print("="*60)
    print("Testing Self-Attention")
    print("="*60)

    # Test configuration
    batch_size, num_heads, seq_len, head_dim = 2, 2, 8, 16
    scale = head_dim ** -0.5

    # Generate random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # PyTorch reference
    with torch.no_grad():
        ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    print(f"Input shape: ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Reference output (first element): {ref_output.flatten()[0].item():.6f}")

    # Save test data
    np.save("test_q.npy", q.numpy())
    np.save("test_k.npy", k.numpy())
    np.save("test_v.npy", v.numpy())
    np.save("test_ref.npy", ref_output.numpy())

    print("\n✓ Test data saved to test_*.npy")
    print(f"  - test_q.npy: {q.numpy().shape}")
    print(f"  - test_k.npy: {k.numpy().shape}")
    print(f"  - test_v.npy: {v.numpy().shape}")
    print(f"  - test_ref.npy: {ref_output.numpy().shape} (reference)")

    return ref_output.numpy()

def test_cross_attention():
    """Test cross-attention with different seq lengths"""
    print("\n" + "="*60)
    print("Testing Cross-Attention")
    print("="*60)

    # Test configuration
    batch_size, num_heads, q_len, kv_len, head_dim = 2, 2, 8, 16, 16
    scale = head_dim ** -0.5

    # Generate random inputs
    q = torch.randn(batch_size, num_heads, q_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim, dtype=torch.float32)

    # Manual computation: Q @ K^T / sqrt(d) @ V
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(attn_weights, v)

    print(f"Query shape: ({batch_size}, {num_heads}, {q_len}, {head_dim})")
    print(f"Key/Value shape: ({batch_size}, {num_heads}, {kv_len}, {head_dim})")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Reference output (first element): {ref_output.flatten()[0].item():.6f}")

    # Save test data
    np.save("test_cross_q.npy", q.numpy())
    np.save("test_cross_k.npy", k.numpy())
    np.save("test_cross_v.npy", v.numpy())
    np.save("test_cross_ref.npy", ref_output.numpy())

    print("\n✓ Cross-attention test data saved")
    print(f"  - test_cross_q.npy: {q.numpy().shape}")
    print(f"  - test_cross_k.npy: {k.numpy().shape}")
    print(f"  - test_cross_v.npy: {v.numpy().shape}")
    print(f"  - test_cross_ref.npy: {ref_output.numpy().shape} (reference)")

    return ref_output.numpy()

def compare_outputs(cpp_output_path, reference, rtol=1e-3, atol=1e-5):
    """Compare C++ output with PyTorch reference"""
    try:
        cpp_output = np.load(cpp_output_path)

        if cpp_output.shape != reference.shape:
            print(f"  ✗ Shape mismatch: {cpp_output.shape} vs {reference.shape}")
            return False

        abs_diff = np.abs(cpp_output - reference)
        max_abs_error = np.max(abs_diff)

        passed = np.all(abs_diff <= atol + rtol * np.abs(reference))

        if passed:
            print(f"  ✓ PASSED - Max abs error: {max_abs_error:.2e}")
        else:
            print(f"  ✗ FAILED - Max abs error: {max_abs_error:.2e}")

        return passed

    except FileNotFoundError:
        print(f"  ⚠ C++ output not found: {cpp_output_path}")
        print(f"    Run C++ test first to generate output")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Quick Attention Validation Test")
    print("="*60 + "\n")

    # Generate test data
    self_ref = test_self_attention()
    cross_ref = test_cross_attention()

    # Check if C++ outputs exist
    print("\n" + "="*60)
    print("Checking C++ Outputs")
    print("="*60 + "\n")

    print("Self-Attention:")
    self_passed = compare_outputs("cpp_self_attention_output.npy", self_ref)

    print("\nCross-Attention:")
    cross_passed = compare_outputs("cpp_cross_attention_output.npy", cross_ref)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if self_passed and cross_passed:
        print("✓ All tests PASSED")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED or not run")
        print("\nTo run C++ tests:")
        print("  1. Save this code as C++ test program")
        print("  2. Compile and run to generate cpp_*_output.npy")
        sys.exit(1)
