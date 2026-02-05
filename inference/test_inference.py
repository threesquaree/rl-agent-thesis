"""
Quick test script for model inference
"""

from test_model import test_single_message

print("=" * 60)
print("Testing Model Inference")
print("=" * 60)

# Test SMDP model
print("\n1. Testing SMDP Model (H3_SMDP_StateMachine.pt):")
try:
    result = test_single_message(
        'pretrained_models/H3_SMDP_StateMachine.pt',
        'What is this painting?',
        'King_Caspar'
    )
    print(f"   ✓ Action: {result['action']}")
    print(f"   ✓ Option: {result['option']}")
    print(f"   ✓ Subaction: {result['subaction']}")
    print(f"   ✓ State shape: {result['state_vector'].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test MDP model
print("\n2. Testing MDP Model (H3_MDP_StateMachine.pt):")
try:
    result = test_single_message(
        'pretrained_models/H3_MDP_StateMachine.pt',
        'Tell me about this artwork',
        'Turban'
    )
    print(f"   ✓ Action: {result['action']}")
    print(f"   ✓ Option: {result['option']}")
    print(f"   ✓ Subaction: {result['subaction']}")
    print(f"   ✓ State shape: {result['state_vector'].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
