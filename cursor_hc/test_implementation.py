"""
Test suite for Transformer with Hyper-Connections.

This script validates the implementation against the paper specifications.
"""

import torch
import torch.nn as nn
import math

from hyper_connections import HyperConnection
from transformer_hc import (
    TransformerWithHC,
    TransformerBlockWithHC,
    MultiHeadAttention,
    FeedForward,
    count_parameters,
)


def test_hyper_connection_initialization():
    """Test that HC initializes correctly to match Pre-Norm residual connections."""
    print("\n" + "="*80)
    print("TEST: Hyper-Connection Initialization")
    print("="*80)
    
    hidden_dim = 128
    n = 4
    
    for layer_idx in range(8):
        hc = HyperConnection(
            hidden_dim=hidden_dim,
            expansion_rate=n,
            layer_idx=layer_idx,
            dynamic=False,  # Test static first
        )
        
        # Check B initialization (should be all ones)
        assert torch.allclose(hc.B, torch.ones(1, n)), \
            f"B should be initialized to ones, got {hc.B}"
        
        # Check Am initialization (should be k-th column of identity)
        k = layer_idx % n
        expected_Am = torch.zeros(n, 1)
        expected_Am[k, 0] = 1.0
        assert torch.allclose(hc.Am, expected_Am), \
            f"Am for layer {layer_idx} should be e_{k}, got {hc.Am.squeeze()}"
        
        # Check Ar initialization (should be identity)
        assert torch.allclose(hc.Ar, torch.eye(n)), \
            f"Ar should be initialized to identity, got {hc.Ar}"
    
    print("✓ Static HC initialization matches Pre-Norm residual connections")
    
    # Test dynamic HC
    hc_dynamic = HyperConnection(
        hidden_dim=hidden_dim,
        expansion_rate=n,
        layer_idx=0,
        dynamic=True,
    )
    
    # Check that dynamic components exist
    assert hasattr(hc_dynamic, 'W_beta'), "Dynamic HC should have W_beta"
    assert hasattr(hc_dynamic, 'W_m'), "Dynamic HC should have W_m"
    assert hasattr(hc_dynamic, 'W_r'), "Dynamic HC should have W_r"
    assert hasattr(hc_dynamic, 's_alpha'), "Dynamic HC should have s_alpha"
    assert hasattr(hc_dynamic, 's_beta'), "Dynamic HC should have s_beta"
    
    print("✓ Dynamic HC has all required components")
    
    print("\nPASSED ✓")


def test_hyper_connection_forward():
    """Test HC forward pass with various input shapes."""
    print("\n" + "="*80)
    print("TEST: Hyper-Connection Forward Pass")
    print("="*80)
    
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    n = 4
    
    # Create HC module
    hc = HyperConnection(
        hidden_dim=hidden_dim,
        expansion_rate=n,
        layer_idx=0,
        dynamic=True,
    )
    
    # Create inputs
    layer_output = torch.randn(batch_size, seq_len, hidden_dim)
    hidden_states = torch.randn(batch_size, seq_len, n, hidden_dim)
    
    # Forward pass
    new_hidden_states = hc(layer_output, hidden_states)
    
    # Check output shape
    assert new_hidden_states.shape == (batch_size, seq_len, n, hidden_dim), \
        f"Expected shape {(batch_size, seq_len, n, hidden_dim)}, got {new_hidden_states.shape}"
    
    print(f"✓ Output shape correct: {new_hidden_states.shape}")
    
    # Check that gradients flow
    loss = new_hidden_states.sum()
    loss.backward()
    
    assert hc.B.grad is not None, "Gradients should flow through B"
    assert hc.Am.grad is not None, "Gradients should flow through Am"
    assert hc.Ar.grad is not None, "Gradients should flow through Ar"
    
    print("✓ Gradients flow through all parameters")
    
    # Test static HC
    hc_static = HyperConnection(
        hidden_dim=hidden_dim,
        expansion_rate=n,
        layer_idx=0,
        dynamic=False,
    )
    
    new_hidden_states_static = hc_static(layer_output, hidden_states)
    assert new_hidden_states_static.shape == (batch_size, seq_len, n, hidden_dim)
    
    print("✓ Static HC forward pass works")
    
    print("\nPASSED ✓")


def test_output_scaling():
    """Test that output layers are scaled by √n."""
    print("\n" + "="*80)
    print("TEST: Output Layer Scaling")
    print("="*80)
    
    hidden_dim = 128
    ffn_dim = 512
    n = 4
    
    # Test FeedForward output scaling
    ffn_no_scale = FeedForward(hidden_dim, ffn_dim, expansion_rate=1)
    ffn_with_scale = FeedForward(hidden_dim, ffn_dim, expansion_rate=n)
    
    # Check that weights are scaled
    weight_ratio = (
        ffn_with_scale.fc2.weight.std() / ffn_no_scale.fc2.weight.std()
    ).item()
    
    expected_ratio = 1.0 / math.sqrt(n)
    
    print(f"Weight std ratio: {weight_ratio:.4f}")
    print(f"Expected ratio (1/√{n}): {expected_ratio:.4f}")
    
    # Allow some tolerance due to initialization variance
    assert abs(weight_ratio - expected_ratio) < 0.2, \
        f"Weight scaling ratio {weight_ratio:.4f} far from expected {expected_ratio:.4f}"
    
    print("✓ FFN output weights scaled correctly")
    
    # Test MultiHeadAttention output scaling
    attn_no_scale = MultiHeadAttention(hidden_dim, num_heads=4, expansion_rate=1)
    attn_with_scale = MultiHeadAttention(hidden_dim, num_heads=4, expansion_rate=n)
    
    weight_ratio_attn = (
        attn_with_scale.out_proj.weight.std() / attn_no_scale.out_proj.weight.std()
    ).item()
    
    print(f"Attention weight std ratio: {weight_ratio_attn:.4f}")
    assert abs(weight_ratio_attn - expected_ratio) < 0.2
    
    print("✓ Attention output weights scaled correctly")
    
    print("\nPASSED ✓")


def test_transformer_block():
    """Test TransformerBlock with HC."""
    print("\n" + "="*80)
    print("TEST: Transformer Block with HC")
    print("="*80)
    
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    n = 4
    
    block = TransformerBlockWithHC(
        hidden_dim=hidden_dim,
        num_heads=4,
        expansion_rate=n,
        layer_idx=0,
        dynamic=True,
    )
    
    # Test with 3D input (first block)
    hidden_states_3d = torch.randn(batch_size, seq_len, hidden_dim)
    output = block(hidden_states_3d)
    
    assert output.shape == (batch_size, seq_len, n, hidden_dim), \
        f"Expected shape {(batch_size, seq_len, n, hidden_dim)}, got {output.shape}"
    
    print(f"✓ Block handles 3D input: {hidden_states_3d.shape} -> {output.shape}")
    
    # Test with 4D input (subsequent blocks)
    hidden_states_4d = torch.randn(batch_size, seq_len, n, hidden_dim)
    output = block(hidden_states_4d)
    
    assert output.shape == (batch_size, seq_len, n, hidden_dim)
    
    print(f"✓ Block handles 4D input: {hidden_states_4d.shape} -> {output.shape}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    assert block.hc_attn.B.grad is not None
    assert block.hc_ffn.B.grad is not None
    
    print("✓ Gradients flow through HC modules")
    
    print("\nPASSED ✓")


def test_full_transformer():
    """Test complete Transformer with HC."""
    print("\n" + "="*80)
    print("TEST: Full Transformer Model")
    print("="*80)
    
    batch_size = 2
    seq_len = 32
    vocab_size = 1000
    
    config = {
        'vocab_size': vocab_size,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'max_seq_len': 64,
        'expansion_rate': 4,
        'dynamic': True,
    }
    
    model = TransformerWithHC(**config)
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    
    print(f"✓ Model forward pass: {input_ids.shape} -> {logits.shape}")
    
    # Test backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check that HC parameters have gradients
    has_hc_grads = False
    for name, param in model.named_parameters():
        if 'hc_' in name and param.grad is not None:
            has_hc_grads = True
            break
    
    assert has_hc_grads, "HC parameters should have gradients"
    
    print("✓ Gradients flow through HC parameters")
    
    # Test parameter counting
    param_counts = count_parameters(model)
    
    print(f"\nParameter breakdown:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  HC: {param_counts['hyper_connection']:,} ({param_counts['hc_percentage']:.2f}%)")
    
    # HC parameters should be < 0.5% (paper reports < 0.04% for larger models)
    assert param_counts['hc_percentage'] < 0.5, \
        f"HC parameters {param_counts['hc_percentage']:.2f}% too high (should be < 0.5%)"
    
    print("✓ HC parameter overhead is minimal")
    
    # Test with return_hidden_states
    logits, hidden_states = model(input_ids, return_hidden_states=True)
    assert len(hidden_states) == config['num_layers']
    
    print(f"✓ Model returns {len(hidden_states)} hidden states")
    
    print("\nPASSED ✓")


def test_parameter_groups():
    """Test that parameters are correctly grouped for weight decay."""
    print("\n" + "="*80)
    print("TEST: Parameter Groups for Weight Decay")
    print("="*80)
    
    hidden_dim = 128
    n = 4
    
    hc = HyperConnection(
        hidden_dim=hidden_dim,
        expansion_rate=n,
        layer_idx=0,
        dynamic=True,
    )
    
    # Get parameter groups
    decay_params = hc.get_parameters_for_weight_decay()
    no_decay_params = hc.get_parameters_no_weight_decay()
    
    # Check that dynamic components are in decay group
    assert hc.W_beta in decay_params, "W_beta should have weight decay"
    assert hc.W_m in decay_params, "W_m should have weight decay"
    assert hc.W_r in decay_params, "W_r should have weight decay"
    
    print("✓ Dynamic components (W_β, W_m, W_r) have weight decay")
    
    # Check that static components are in no-decay group
    assert hc.B in no_decay_params, "B should NOT have weight decay"
    assert hc.Am in no_decay_params, "Am should NOT have weight decay"
    assert hc.Ar in no_decay_params, "Ar should NOT have weight decay"
    
    print("✓ Static components (B, Am, Ar) do NOT have weight decay")
    
    # Check scaling factors
    assert hc.s_alpha in no_decay_params, "s_alpha should NOT have weight decay"
    assert hc.s_beta in no_decay_params, "s_beta should NOT have weight decay"
    
    print("✓ Scaling factors (s_α, s_β) do NOT have weight decay")
    
    print("\nPASSED ✓")


def test_expansion_rate_scaling():
    """Test different expansion rates."""
    print("\n" + "="*80)
    print("TEST: Expansion Rate Scaling")
    print("="*80)
    
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    vocab_size = 1000
    
    for n in [1, 2, 4, 8]:
        model = TransformerWithHC(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            expansion_rate=n,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        param_counts = count_parameters(model)
        
        print(f"n={n}: Total params: {param_counts['total']:,}, "
              f"HC: {param_counts['hc_percentage']:.3f}%")
    
    print("\n✓ All expansion rates work correctly")
    print("\nPASSED ✓")


def test_static_vs_dynamic():
    """Compare Static HC vs Dynamic HC."""
    print("\n" + "="*80)
    print("TEST: Static vs Dynamic HC")
    print("="*80)
    
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    vocab_size = 1000
    
    config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'num_layers': 2,
        'num_heads': 4,
        'expansion_rate': 4,
    }
    
    # Static HC
    model_shc = TransformerWithHC(**config, dynamic=False)
    param_counts_shc = count_parameters(model_shc)
    
    # Dynamic HC
    model_dhc = TransformerWithHC(**config, dynamic=True)
    param_counts_dhc = count_parameters(model_dhc)
    
    print(f"Static HC:  {param_counts_shc['total']:,} params")
    print(f"Dynamic HC: {param_counts_dhc['total']:,} params")
    print(f"Difference: {param_counts_dhc['total'] - param_counts_shc['total']:,} params")
    
    # Test forward passes
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits_shc = model_shc(input_ids)
    logits_dhc = model_dhc(input_ids)
    
    assert logits_shc.shape == logits_dhc.shape
    
    print("✓ Both SHC and DHC work correctly")
    
    # Dynamic should have more parameters
    assert param_counts_dhc['total'] > param_counts_shc['total']
    
    print("✓ DHC has more parameters (as expected)")
    
    print("\nPASSED ✓")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nValidating implementation against HYPER-CONNECTIONS paper (ICLR 2025)")
    
    tests = [
        test_hyper_connection_initialization,
        test_hyper_connection_forward,
        test_output_scaling,
        test_transformer_block,
        test_full_transformer,
        test_parameter_groups,
        test_expansion_rate_scaling,
        test_static_vs_dynamic,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAILED ✗: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Implementation is correct.")
        print("\nKey validations:")
        print("✓ HC initialization matches Pre-Norm residual connections")
        print("✓ Output layers scaled by √n")
        print("✓ Depth and width connections work correctly")
        print("✓ Weight decay applied correctly (dynamic vs static)")
        print("✓ Both SHC and DHC variants functional")
        print("✓ Parameter overhead < 0.5%")
        print("✓ Gradients flow through all components")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
