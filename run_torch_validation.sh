#!/bin/bash
# Complete PyTorch validation test runner

set -e

echo "=================================================="
echo "  PyTorch Validation Test Runner"
echo "=================================================="
echo ""

# Step 1: Generate test data with PyTorch
echo "Step 1: Generating test data with PyTorch..."
python3 torch_validation.py

if [ ! -d "test_data" ]; then
    echo "ERROR: test_data directory not created"
    exit 1
fi

echo "✓ Test data generated successfully"
echo ""

# Step 2: Build C++ validation test
echo "Step 2: Building C++ validation test..."
cd tensor_cpp
make torch-validation

if [ ! -f "build/torch_validation" ]; then
    echo "ERROR: torch_validation binary not built"
    exit 1
fi

echo "✓ C++ validation test built successfully"
cd ..
echo ""

# Step 3: Run C++ validation test
echo "Step 3: Running C++ validation test..."
cd tensor_cpp
./build/torch_validation
cd ..

echo "✓ C++ validation test completed"
echo ""

# Step 4: Check results
echo "Step 4: Checking results..."
python3 torch_validation.py --check-results

echo ""
echo "=================================================="
echo "  Validation Complete!"
echo "=================================================="
