"""
Test script to verify the Hyper-Connections implementation

This script runs basic tests to ensure the implementation is working correctly.
"""

import torch
import sys


def test_hyper_connection_block():
    """Test the HyperConnection block"""
    print("Testing HyperConnection block...")
    
    from hyper_connections import HyperConnection
    
    hidden_size = 768
    expansion_rate = 4
    batch_size = 2
    seq_len = 128
    
    # Test DHC (Dynamic Hyper-Connections)
    hc_dhc = HyperConnection(
        hidden_size=hidden_size,
        expansion_rate=expansion_rate,
        use_tanh=True,
        static_weights=False
    )
    
    # Test forward pass
    layer_output = torch.randn(batch_size, seq_len, hidden_size)
    input_hidden = torch.randn(batch_size, seq_len, hidden_size)
    
    output = hc_dhc(layer_output, input_hidden)
    
    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {output.shape}"
    
    # Test that alpha and beta are learnable
    assert hc_dhc.alpha.requires_grad, "Alpha should be learnable"
    assert hc_dhc.beta.requires_grad, "Beta should be learnable"
    
    # Test SHC (Static Hyper-Connections)
    hc_shc = HyperConnection(
        hidden_size=hidden_size,
        expansion_rate=expansion_rate,
        use_tanh=False,
        static_weights=True
    )
    
    output_shc = hc_shc(layer_output, input_hidden)
    assert output_shc.shape == (batch_size, seq_len, hidden_size), \
        "SHC output shape mismatch"
    
    # Test that alpha and beta are NOT learnable in SHC
    assert not hc_shc.alpha.requires_grad, "Alpha should be static in SHC"
    assert not hc_shc.beta.requires_grad, "Beta should be static in SHC"
    
    print("✓ HyperConnection block tests passed!")
    return True


def test_transformer_layer():
    """Test TransformerLayerWithHC"""
    print("\nTesting TransformerLayerWithHC...")
    
    from hyper_connections import TransformerLayerWithHC
    
    hidden_size = 768
    num_heads = 12
    ffn_hidden_size = 3072
    expansion_rate = 4
    batch_size = 2
    seq_len = 128
    
    layer = TransformerLayerWithHC(
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_hidden_size=ffn_hidden_size,
        expansion_rate=expansion_rate,
        use_tanh=True,
        static_weights=False
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {output.shape}"
    
    # Test with attention mask
    attention_mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    ).bool()
    output_masked = layer(x, attention_mask=attention_mask)
    
    assert output_masked.shape == (batch_size, seq_len, hidden_size), \
        "Output shape with attention mask mismatch"
    
    print("✓ TransformerLayerWithHC tests passed!")
    return True


def test_transformer_model():
    """Test full TransformerWithHC model"""
    print("\nTesting TransformerWithHC model...")
    
    from hyper_connections import TransformerWithHC
    
    vocab_size = 1000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    expansion_rate = 4
    batch_size = 2
    seq_len = 64
    
    model = TransformerWithHC(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion_rate=expansion_rate,
        use_tanh=True,
    )
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {logits.shape}"
    
    # Test parameter counting
    num_params = model.get_num_params()
    assert num_params > 0, "Model should have parameters"
    
    num_params_no_embed = model.get_num_params(non_embedding=True)
    # Note: Due to weight tying between embeddings and LM head,
    # non_embedding count excludes embeddings but the LM head shares those weights
    assert num_params_no_embed <= num_params, \
        "Non-embedding params should be less than or equal to total params"
    
    print(f"  Model has {num_params:,} total parameters")
    print(f"  Model has {num_params_no_embed:,} non-embedding parameters")
    
    print("✓ TransformerWithHC model tests passed!")
    return True


def test_backward_pass():
    """Test that gradients flow correctly"""
    print("\nTesting backward pass and gradient flow...")
    
    from hyper_connections import TransformerWithHC
    
    model = TransformerWithHC(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        expansion_rate=4,
    )
    
    # Forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    target_ids = torch.randint(0, 1000, (2, 32))
    
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check that hyper-connection parameters have gradients
    first_layer = model.layers[0]
    
    assert first_layer.attention_hc.alpha.grad is not None, \
        "Alpha should have gradients"
    assert first_layer.attention_hc.beta.grad is not None, \
        "Beta should have gradients"
    
    # Check gradient magnitudes
    alpha_grad_norm = first_layer.attention_hc.alpha.grad.norm().item()
    beta_grad_norm = first_layer.attention_hc.beta.grad.norm().item()
    
    assert alpha_grad_norm > 0, "Alpha gradients should be non-zero"
    assert beta_grad_norm > 0, "Beta gradients should be non-zero"
    
    print(f"  Alpha gradient norm: {alpha_grad_norm:.6f}")
    print(f"  Beta gradient norm: {beta_grad_norm:.6f}")
    
    print("✓ Backward pass tests passed!")
    return True


def test_different_expansion_rates():
    """Test different expansion rates"""
    print("\nTesting different expansion rates...")
    
    from hyper_connections import TransformerWithHC
    
    expansion_rates = [1, 2, 4, 8]
    
    for n in expansion_rates:
        model = TransformerWithHC(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            expansion_rate=n,
        )
        
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000), \
            f"Expansion rate {n}: output shape mismatch"
        
        print(f"  ✓ Expansion rate n={n} works correctly")
    
    print("✓ Different expansion rates tests passed!")
    return True


def test_dhc_vs_shc():
    """Test DHC vs SHC variants"""
    print("\nTesting DHC vs SHC variants...")
    
    from hyper_connections import TransformerWithHC
    
    config = {
        'vocab_size': 1000,
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 4,
        'expansion_rate': 4,
    }
    
    # Test DHC
    model_dhc = TransformerWithHC(**config, static_weights=False)
    input_ids = torch.randint(0, 1000, (2, 32))
    logits_dhc = model_dhc(input_ids)
    
    # Test SHC
    model_shc = TransformerWithHC(**config, static_weights=True)
    logits_shc = model_shc(input_ids)
    
    assert logits_dhc.shape == logits_shc.shape, \
        "DHC and SHC should have same output shape"
    
    # Check that DHC has learnable params but SHC doesn't
    dhc_layer = model_dhc.layers[0]
    shc_layer = model_shc.layers[0]
    
    assert dhc_layer.attention_hc.alpha.requires_grad, \
        "DHC alpha should be learnable"
    assert not shc_layer.attention_hc.alpha.requires_grad, \
        "SHC alpha should NOT be learnable"
    
    print("  ✓ DHC variant works correctly")
    print("  ✓ SHC variant works correctly")
    print("✓ DHC vs SHC tests passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Hyper-Connections Implementation Tests")
    print("=" * 70)
    
    tests = [
        test_hyper_connection_block,
        test_transformer_layer,
        test_transformer_model,
        test_backward_pass,
        test_different_expansion_rates,
        test_dhc_vs_shc,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test {test.__name__} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
