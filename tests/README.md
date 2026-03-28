# Tests

Unit and integration tests for the implementation.

## Test Structure

```
tests/
├── test_coupling.py       # Test LearnableCouplingConstants
├── test_losses.py         # Test loss functions
├── test_ieee_cases.py     # Test IEEE case loading
├── test_stability.py      # Test stability margin computation
└── test_integration.py    # End-to-end integration tests
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```
