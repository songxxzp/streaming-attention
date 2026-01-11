#!/usr/bin/env python3
"""
PyTorch Validation Tests for tensor_cpp Operators

This script generates test data, computes reference results using PyTorch,
and validates the C++ implementations against these golden references.
"""

import torch
import numpy as np
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class TestCase:
    """A single test case with input data and reference output"""
    name: str
    inputs: Dict[str, np.ndarray]
    reference_output: np.ndarray
    tolerance: Dict[str, float] = None

    def __post_init__(self):
        if self.tolerance is None:
            self.tolerance = {'rtol': 1e-3, 'atol': 1e-5}


class PyTorchValidator:
    """Generate reference outputs using PyTorch implementations"""

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0):
        """
        Reference implementation using PyTorch's F.scaled_dot_product_attention
        Q, K, V: [batch, heads, seq_len, head_dim]
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
        )

    @staticmethod
    def manual_scaled_dot_product_attention(query, key, value, scale=None):
        """
        Manual implementation for verification
        Q, K, V: [batch, heads, seq_len, head_dim]
        """
        if scale is None:
            scale = query.size(-1) ** -0.5

        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply to values: attn_weights @ V
        output = torch.matmul(attn_weights, value)
        return output

    @staticmethod
    def linear(input, weight, bias=None):
        """Linear layer: y = xA^T + b"""
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def rms_norm(input, weight=None, eps=1e-8):
        """RMS Normalization"""
        squared_mean = torch.mean(input * input, dim=-1, keepdim=True)
        rms = torch.sqrt(squared_mean + eps)
        output = input / rms
        if weight is not None:
            output = output * weight
        return output

    @staticmethod
    def swiglu(x, gate):
        """SwiGLU activation"""
        return torch.nn.functional.silu(gate) * x

    @staticmethod
    def apply_rope(x, cos, sin):
        """Apply rotary position embedding"""
        # x: [batch, seq_len, heads, head_dim]
        # cos, sin: [seq_len, head_dim//2]

        # Split into real and imaginary parts
        x1 = x[..., :x.size(-1)//2]  # [batch, seq_len, heads, head_dim//2]
        x2 = x[..., x.size(-1)//2:]  # [batch, seq_len, heads, head_dim//2]

        # Apply rotation
        # [batch, seq_len, heads, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rotated

    @staticmethod
    def embedding(indices, weight, padding_idx=-1):
        """Embedding lookup"""
        return torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)

    @staticmethod
    def argmax(input, dim=-1, keepdim=False):
        """Argmax"""
        return torch.argmax(input, dim=dim, keepdim=keepdim)


class TestGenerator:
    """Generate test cases for all operators"""

    def __init__(self, output_dir: str = "test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_cases: List[TestCase] = []

    def save_array(self, name: str, data: np.ndarray) -> str:
        """Save array to .npy file and return path"""
        path = self.output_dir / f"{name}.npy"
        np.save(path, data)
        return str(path)

    def save_test_case(self, test_case: TestCase):
        """Save test case to JSON and arrays"""
        # Save metadata
        metadata = {
            'name': test_case.name,
            'tolerance': test_case.tolerance,
            'inputs': {},
            'reference_output': test_case.reference_output.shape,
        }

        # Save input arrays
        for key, value in test_case.inputs.items():
            path = self.save_array(f"{test_case.name}_{key}", value)
            metadata['inputs'][key] = {
                'path': path,
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            }

        # Save reference output
        ref_path = self.save_array(f"{test_case.name}_ref", test_case.reference_output)
        metadata['reference_output_path'] = ref_path

        # Save metadata
        meta_path = self.output_dir / f"{test_case.name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_self_attention_tests(self):
        """Test self-attention with various configurations"""
        print("Generating self-attention tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'heads': 2, 'seq': 8, 'dim': 16},
            {'batch': 1, 'heads': 4, 'seq': 16, 'dim': 32},
            {'batch': 4, 'heads': 8, 'seq': 64, 'dim': 64},
        ]

        for i, config in enumerate(configs):
            batch, heads, seq, dim = config['batch'], config['heads'], config['seq'], config['dim']
            scale = dim ** -0.5

            # Generate random inputs
            q = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
            k = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
            v = torch.randn(batch, heads, seq, dim, dtype=torch.float32)

            # Compute reference using PyTorch
            ref = validator.scaled_dot_product_attention(q, k, v)

            # Create test case
            test_case = TestCase(
                name=f"self_attention_{i+1}",
                inputs={
                    'query': q.numpy(),
                    'key': k.numpy(),
                    'value': v.numpy(),
                    'scale': np.array(scale, dtype=np.float32),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-3, 'atol': 1e-4}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_cross_attention_tests(self):
        """Test cross-attention with different seq lengths"""
        print("Generating cross-attention tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'heads': 2, 'q_len': 8, 'kv_len': 16, 'dim': 16},
            {'batch': 1, 'heads': 4, 'q_len': 32, 'kv_len': 128, 'dim': 32},
        ]

        for i, config in enumerate(configs):
            batch, heads, q_len, kv_len, dim = config['batch'], config['heads'], config['q_len'], config['kv_len'], config['dim']
            scale = dim ** -0.5

            q = torch.randn(batch, heads, q_len, dim, dtype=torch.float32)
            k = torch.randn(batch, heads, kv_len, dim, dtype=torch.float32)
            v = torch.randn(batch, heads, kv_len, dim, dtype=torch.float32)

            ref = validator.manual_scaled_dot_product_attention(q, k, v, scale)

            test_case = TestCase(
                name=f"cross_attention_{i+1}",
                inputs={
                    'query': q.numpy(),
                    'key': k.numpy(),
                    'value': v.numpy(),
                    'scale': np.array(scale, dtype=np.float32),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-3, 'atol': 1e-4}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_linear_tests(self):
        """Test linear layer"""
        print("Generating linear layer tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'in_features': 64, 'out_features': 32},
            {'batch': 4, 'in_features': 128, 'out_features': 64},
        ]

        for i, config in enumerate(configs):
            batch, in_feat, out_feat = config['batch'], config['in_features'], config['out_features']

            x = torch.randn(batch, in_feat, dtype=torch.float32)
            weight = torch.randn(out_feat, in_feat, dtype=torch.float32)
            bias = torch.randn(out_feat, dtype=torch.float32)

            ref = validator.linear(x, weight, bias)

            test_case = TestCase(
                name=f"linear_{i+1}",
                inputs={
                    'input': x.numpy(),
                    'weight': weight.numpy(),
                    'bias': bias.numpy(),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-4, 'atol': 1e-5}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_rms_norm_tests(self):
        """Test RMS normalization"""
        print("Generating RMS norm tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'seq': 16, 'hidden': 64},
            {'batch': 4, 'seq': 32, 'hidden': 128},
        ]

        for i, config in enumerate(configs):
            batch, seq, hidden = config['batch'], config['seq'], config['hidden']

            x = torch.randn(batch, seq, hidden, dtype=torch.float32)
            weight = torch.randn(hidden, dtype=torch.float32)

            ref = validator.rms_norm(x, weight)

            test_case = TestCase(
                name=f"rms_norm_{i+1}",
                inputs={
                    'input': x.numpy(),
                    'weight': weight.numpy(),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-4, 'atol': 1e-5}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_embedding_tests(self):
        """Test embedding lookup"""
        print("Generating embedding tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'seq': 10, 'num_embeddings': 100, 'embedding_dim': 64},
            {'batch': 4, 'seq': 32, 'num_embeddings': 1000, 'embedding_dim': 128},
        ]

        for i, config in enumerate(configs):
            batch, seq, num_emb, emb_dim = config['batch'], config['seq'], config['num_embeddings'], config['embedding_dim']

            indices = torch.randint(0, num_emb, (batch, seq))
            weight = torch.randn(num_emb, emb_dim, dtype=torch.float32)

            ref = validator.embedding(indices, weight)

            test_case = TestCase(
                name=f"embedding_{i+1}",
                inputs={
                    'indices': indices.numpy().astype(np.int64),
                    'weight': weight.numpy(),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-5, 'atol': 1e-6}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_argmax_tests(self):
        """Test argmax"""
        print("Generating argmax tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'seq': 10},
            {'batch': 4, 'vocab': 1000},
        ]

        for i, config in enumerate(configs):
            batch, size = config['batch'], config.get('seq', config.get('vocab'))

            x = torch.randn(batch, size, dtype=torch.float32)
            ref = validator.argmax(x, dim=-1)

            test_case = TestCase(
                name=f"argmax_{i+1}",
                inputs={
                    'input': x.numpy(),
                },
                reference_output=ref.numpy().astype(np.int64),
                tolerance={'rtol': 0, 'atol': 0}  # Exact match for integer indices
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_swiglu_tests(self):
        """Test SwiGLU activation"""
        print("Generating SwiGLU tests...")

        validator = PyTorchValidator()
        configs = [
            {'batch': 2, 'seq': 16, 'hidden': 64},
            {'batch': 4, 'seq': 32, 'hidden': 128},
        ]

        for i, config in enumerate(configs):
            batch, seq, hidden = config['batch'], config['seq'], config['hidden']

            x = torch.randn(batch, seq, hidden, dtype=torch.float32)
            gate = torch.randn(batch, seq, hidden, dtype=torch.float32)

            ref = validator.swiglu(x, gate)

            test_case = TestCase(
                name=f"swiglu_{i+1}",
                inputs={
                    'x': x.numpy(),
                    'gate': gate.numpy(),
                },
                reference_output=ref.numpy(),
                tolerance={'rtol': 1e-5, 'atol': 1e-6}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_streaming_attention_tests(self):
        """Test streaming attention (single query format)"""
        print("Generating streaming attention tests...")

        configs = [
            {'T': 512, 'd': 64},
            {'T': 1024, 'd': 128},
            {'T': 2048, 'd': 256},
        ]

        for i, config in enumerate(configs):
            T, d = config['T'], config['d']

            # Q: [d], K: [T, d], V: [T, d]
            Q = torch.randn(d, dtype=torch.float32)
            K = torch.randn(T, d, dtype=torch.float32)
            V = torch.randn(T, d, dtype=torch.float32)

            # Compute reference: attention(Q, K, V) where Q is 1xd, K is Txd
            # scores = Q @ K^T -> [T]
            scores = torch.matmul(Q.unsqueeze(0), K.T)  # [1, T]
            attn_weights = torch.softmax(scores, dim=-1)  # [1, T]
            output = torch.matmul(attn_weights, V)  # [1, d]
            output = output.squeeze(0)  # [d]

            test_case = TestCase(
                name=f"streaming_attention_{i+1}",
                inputs={
                    'Q': Q.numpy(),
                    'K': K.numpy(),
                    'V': V.numpy(),
                    'T': np.array(T, dtype=np.int32),
                    'd': np.array(d, dtype=np.int32),
                },
                reference_output=output.numpy(),
                tolerance={'rtol': 1e-3, 'atol': 1e-4}
            )
            self.save_test_case(test_case)
            self.test_cases.append(test_case)
            print(f"  ✓ Generated test case: {test_case.name}")

    def generate_all_tests(self):
        """Generate all test cases"""
        print("\n" + "="*60)
        print("Generating PyTorch Reference Test Cases")
        print("="*60 + "\n")

        self.generate_self_attention_tests()
        self.generate_cross_attention_tests()
        self.generate_streaming_attention_tests()
        self.generate_linear_tests()
        self.generate_rms_norm_tests()
        self.generate_embedding_tests()
        self.generate_argmax_tests()
        self.generate_swiglu_tests()

        # Save index
        index = {
            'test_data_dir': str(self.output_dir),
            'test_cases': [tc.name for tc in self.test_cases],
            'total': len(self.test_cases)
        }

        with open(self.output_dir / 'index.json', 'w') as f:
            json.dump(index, f, indent=2)

        print(f"\n✓ Generated {len(self.test_cases)} test cases")
        print(f"✓ Test data saved to: {self.output_dir}")


def load_and_validate(test_name: str, test_data_dir: str = "test_data"):
    """Load a test case and return inputs, reference, and tolerance"""
    meta_path = Path(test_data_dir) / f"{test_name}_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Test case not found: {test_name}")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Load inputs
    inputs = {}
    for key, info in metadata['inputs'].items():
        if key in ['T', 'd', 'scale']:
            # Scalar values
            inputs[key] = np.load(info['path']).item()
        else:
            inputs[key] = np.load(info['path'])

    # Load reference
    reference = np.load(metadata['reference_output_path'])

    return inputs, reference, metadata['tolerance']


def compare_outputs(actual: np.ndarray, reference: np.ndarray, tolerance: Dict[str, float]) -> Dict:
    """Compare actual output with reference"""
    if actual.shape != reference.shape:
        return {
            'passed': False,
            'error': f'Shape mismatch: {actual.shape} vs {reference.shape}'
        }

    # Compute errors
    abs_diff = np.abs(actual - reference)
    rel_diff = abs_diff / (np.abs(reference) + tolerance['atol'])

    max_abs_error = np.max(abs_diff)
    max_rel_error = np.max(rel_diff)
    mean_abs_error = np.mean(abs_diff)

    passed = np.all(abs_diff <= tolerance['atol'] + tolerance['rtol'] * np.abs(reference))

    return {
        'passed': passed,
        'max_abs_error': float(max_abs_error),
        'max_rel_error': float(max_rel_error),
        'mean_abs_error': float(mean_abs_error),
        'tolerance': tolerance
    }


def check_results(test_data_dir: str = "test_data"):
    """Check C++ outputs against PyTorch references"""
    print("\n" + "="*60)
    print("Validating C++ Outputs Against PyTorch References")
    print("="*60 + "\n")

    # Load test index
    index_path = Path(test_data_dir) / 'index.json'
    if not index_path.exists():
        print(f"✗ Test index not found: {index_path}")
        print("  Run without arguments first to generate test data")
        return False

    with open(index_path, 'r') as f:
        index = json.load(f)

    passed = 0
    failed = 0
    failed_tests = []

    for test_name in index['test_cases']:
        try:
            inputs, reference, tolerance = load_and_validate(test_name, test_data_dir)

            # Load C++ output
            cpp_output_path = Path(test_data_dir) / f"cpp_{test_name}_output.npy"

            if not cpp_output_path.exists():
                print(f"⚠ {test_name}: C++ output not found (skipped)")
                continue

            cpp_output = np.load(cpp_output_path)

            # Compare
            result = compare_outputs(cpp_output, reference, tolerance)

            if result['passed']:
                print(f"✓ {test_name}: PASSED")
                print(f"    Max abs error: {result['max_abs_error']:.2e}")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                else:
                    print(f"    Max abs error: {result['max_abs_error']:.2e} (tolerance: {tolerance['atol']:.2e})")
                    print(f"    Max rel error: {result['max_rel_error']:.2e} (tolerance: {tolerance['rtol']:.2e})")
                failed += 1
                failed_tests.append(test_name)

        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            failed += 1
            failed_tests.append(test_name)

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total tests:  {passed + failed}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")

    print()

    return failed == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--check-results':
        # Check C++ results
        success = check_results()
        sys.exit(0 if success else 1)
    else:
        # Generate test cases
        generator = TestGenerator(output_dir="test_data")
        generator.generate_all_tests()

        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("1. Build the C++ validation test:")
        print("   cd tensor_cpp && make torch-validation")
        print("\n2. Run the C++ validation test:")
        print("   ./build/torch_validation")
        print("\n3. Check results:")
        print("   python torch_validation.py --check-results")
