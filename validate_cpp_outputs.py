#!/usr/bin/env python3
"""
Validate C++ outputs against PyTorch references

This script loads C++ binary outputs and compares them with PyTorch references.
"""

import torch
import numpy as np
import struct
import os

# Set seed
torch.manual_seed(42)
np.random.seed(42)

def load_binary(filename, dtype=np.float32, shape=None):
    """Load simple binary file (no numpy header)"""
    if not os.path.exists(filename):
        return None

    with open(filename, 'rb') as f:
        data = f.read()

    # Convert bytes to numpy array
    if dtype == np.float32:
        array = np.frombuffer(data, dtype=np.float32)
    elif dtype == np.int64:
        array = np.frombuffer(data, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if shape is not None:
        array = array.reshape(shape)

    return array

def save_binary(filename, data):
    """Save data as simple binary file"""
    with open(filename, 'wb') as f:
        if data.dtype == np.float32:
            f.write(data.tobytes())
        elif data.dtype == np.int64:
            f.write(data.tobytes())
        else:
            raise ValueError(f"Unsupported dtype: {data.dtype}")

def compare_outputs(cpp_output, ref_output, rtol=1e-3, atol=1e-5):
    """Compare C++ output with PyTorch reference"""
    if cpp_output.shape != ref_output.shape:
        return False, f"Shape mismatch: {cpp_output.shape} vs {ref_output.shape}"

    abs_diff = np.abs(cpp_output - ref_output)
    max_abs_error = np.max(abs_diff)

    passed = np.all(abs_diff <= atol + rtol * np.abs(ref_output))
    return passed, max_abs_error

def test_self_attention():
    print("Testing self_attention_1...")

    # Generate or load inputs
    batch, heads, seq, dim = 2, 2, 8, 16
    scale = dim ** -0.5

    # Try to load from binary
    q_data = load_binary("../test_data/self_attention_1_query.bin", shape=(batch, heads, seq, dim))
    k_data = load_binary("../test_data/self_attention_1_key.bin", shape=(batch, heads, seq, dim))
    v_data = load_binary("../test_data/self_attention_1_value.bin", shape=(batch, heads, seq, dim))

    if q_data is None:
        print("  Generating new test data...")
        q = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
        k = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
        v = torch.randn(batch, heads, seq, dim, dtype=torch.float32)

        # Save inputs
        save_binary("../test_data/self_attention_1_query.bin", q.numpy())
        save_binary("../test_data/self_attention_1_key.bin", k.numpy())
        save_binary("../test_data/self_attention_1_value.bin", v.numpy())
    else:
        print("  Loading existing test data...")
        q = torch.from_numpy(q_data).float()
        k = torch.from_numpy(k_data).float()
        v = torch.from_numpy(v_data).float()

    # PyTorch reference
    with torch.no_grad():
        ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Save reference
    save_binary("../test_data/self_attention_1_ref.bin", ref_output.numpy())

    # Load C++ output
    cpp_output = load_binary("../test_data/cpp_self_attention_1_output.bin",
                              shape=ref_output.shape)

    if cpp_output is None:
        print("  ⚠ C++ output not found. Run C++ test first.")
        return False

    # Compare
    passed, error = compare_outputs(cpp_output, ref_output.numpy())

    if passed:
        print(f"  ✓ PASSED - Max abs error: {error:.2e}")
        return True
    else:
        print(f"  ✗ FAILED - Max abs error: {error:.2e}")
        return False

def test_cross_attention():
    print("Testing cross_attention_1...")

    batch, heads, q_len, kv_len, dim = 2, 2, 8, 16, 16
    scale = dim ** -0.5

    # Load or generate
    q_data = load_binary("../test_data/cross_attention_1_query.bin", shape=(batch, heads, q_len, dim))
    k_data = load_binary("../test_data/cross_attention_1_key.bin", shape=(batch, heads, kv_len, dim))
    v_data = load_binary("../test_data/cross_attention_1_value.bin", shape=(batch, heads, kv_len, dim))

    if q_data is None:
        print("  Generating new test data...")
        q = torch.randn(batch, heads, q_len, dim, dtype=torch.float32)
        k = torch.randn(batch, heads, kv_len, dim, dtype=torch.float32)
        v = torch.randn(batch, heads, kv_len, dim, dtype=torch.float32)

        save_binary("../test_data/cross_attention_1_query.bin", q.numpy())
        save_binary("../test_data/cross_attention_1_key.bin", k.numpy())
        save_binary("../test_data/cross_attention_1_value.bin", v.numpy())
    else:
        print("  Loading existing test data...")
        q = torch.from_numpy(q_data).float()
        k = torch.from_numpy(k_data).float()
        v = torch.from_numpy(v_data).float()

    # Manual cross-attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(attn_weights, v)

    # Save reference
    save_binary("../test_data/cross_attention_1_ref.bin", ref_output.numpy())

    # Load C++ output
    cpp_output = load_binary("../test_data/cpp_cross_attention_1_output.bin",
                              shape=ref_output.shape)

    if cpp_output is None:
        print("  ⚠ C++ output not found. Run C++ test first.")
        return False

    # Compare
    passed, error = compare_outputs(cpp_output, ref_output.numpy())

    if passed:
        print(f"  ✓ PASSED - Max abs error: {error:.2e}")
        return True
    else:
        print(f"  ✗ FAILED - Max abs error: {error:.2e}")
        return False

def test_streaming_attention():
    print("Testing streaming_attention_1...")

    T, d = 512, 64

    # Load or generate
    Q_data = load_binary("../test_data/streaming_attention_1_Q.bin", shape=(d,))
    K_data = load_binary("../test_data/streaming_attention_1_K.bin", shape=(T, d))
    V_data = load_binary("../test_data/streaming_attention_1_V.bin", shape=(T, d))

    if Q_data is None:
        print("  Generating new test data...")
        np.random.seed(42)
        Q = np.random.randn(d).astype(np.float32)
        K = np.random.randn(T, d).astype(np.float32)
        V = np.random.randn(T, d).astype(np.float32)

        save_binary("../test_data/streaming_attention_1_Q.bin", Q)
        save_binary("../test_data/streaming_attention_1_K.bin", K)
        save_binary("../test_data/streaming_attention_1_V.bin", V)
    else:
        print("  Loading existing test data...")
        Q = Q_data
        K = K_data
        V = V_data

    # Manual attention for single query
    scores = Q @ K.T  # [T]
    attn_weights = np.exp(scores - np.max(scores))
    attn_weights = attn_weights / np.sum(attn_weights)
    ref_output = attn_weights @ V  # [d]

    # Save reference
    save_binary("../test_data/streaming_attention_1_ref.bin", ref_output)

    # Load C++ output
    cpp_output = load_binary("../test_data/cpp_streaming_attention_1_output.bin",
                              shape=ref_output.shape)

    if cpp_output is None:
        print("  ⚠ C++ output not found. Run C++ test first.")
        return False

    # Compare
    passed, error = compare_outputs(cpp_output, ref_output, rtol=1e-3, atol=1e-4)

    if passed:
        print(f"  ✓ PASSED - Max abs error: {error:.2e}")
        return True
    else:
        print(f"  ✗ FAILED - Max abs error: {error:.2e}")
        return False

def test_linear():
    print("Testing linear_1...")

    batch, in_feat, out_feat = 2, 64, 32

    # Try to load from binary
    x_data = load_binary("../test_data/linear_1_x.bin", shape=(batch, in_feat))
    w_data = load_binary("../test_data/linear_1_weight.bin", shape=(out_feat, in_feat))
    b_data = load_binary("../test_data/linear_1_bias.bin", shape=(out_feat,))

    if x_data is None:
        print("  Generating new test data...")
        # Use same random seed as C++
        np.random.seed(42)
        x = np.random.randn(batch, in_feat).astype(np.float32)
        weight = np.random.randn(out_feat, in_feat).astype(np.float32)
        bias = np.random.randn(out_feat).astype(np.float32)

        # Save for C++
        save_binary("../test_data/linear_1_x.bin", x)
        save_binary("../test_data/linear_1_weight.bin", weight)
        save_binary("../test_data/linear_1_bias.bin", bias)
    else:
        print("  Loading existing test data...")
        x = x_data
        weight = w_data
        bias = b_data

    # PyTorch reference
    x_torch = torch.from_numpy(x.copy())
    weight_torch = torch.from_numpy(weight.copy())
    bias_torch = torch.from_numpy(bias.copy())

    ref_output = torch.nn.functional.linear(x_torch, weight_torch, bias_torch)

    # Save reference
    save_binary("../test_data/linear_1_ref.bin", ref_output.numpy())

    # Load C++ output
    cpp_output = load_binary("../test_data/cpp_linear_1_output.bin",
                              shape=ref_output.shape)

    if cpp_output is None:
        print("  ⚠ C++ output not found. Run C++ test first.")
        return False

    # Compare
    passed, error = compare_outputs(cpp_output, ref_output.numpy(), rtol=1e-4, atol=1e-5)

    if passed:
        print(f"  ✓ PASSED - Max abs error: {error:.2e}")
        return True
    else:
        print(f"  ✗ FAILED - Max abs error: {error:.2e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Validating C++ Outputs Against PyTorch")
    print("="*60 + "\n")

    results = []

    results.append(test_self_attention())
    results.append(test_cross_attention())
    results.append(test_streaming_attention())
    results.append(test_linear())

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if all(results):
        print("\n✓ All tests PASSED!")
        exit(0)
    else:
        print("\n✗ Some tests FAILED")
        exit(1)
