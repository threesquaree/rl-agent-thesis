"""
Comprehensive inference test showing different scenarios
"""

from test_model import test_single_message, test_conversation, load_trained_model
from collections import defaultdict

print("=" * 70)
print("COMPREHENSIVE MODEL INFERENCE TEST")
print("=" * 70)

# Test 1: Single message with SMDP model
print("\n" + "=" * 70)
print("TEST 1: Single Message - SMDP Model")
print("=" * 70)
result = test_single_message(
    'pretrained_models/H3_SMDP_StateMachine.pt',
    'What is this painting?',
    'King_Caspar'
)
print(f"User: 'What is this painting?'")
print(f"Agent Response:")
print(f"  - Action: {result['action']}")
print(f"  - Option: {result['option']}")
print(f"  - Subaction: {result['subaction']}")
print(f"  - State Dimension: {result['state_vector'].shape[0]}")
print(f"  - Terminated Option: {result['action_dict'].get('terminated', False)}")

# Test 2: Single message with MDP model
print("\n" + "=" * 70)
print("TEST 2: Single Message - MDP Model")
print("=" * 70)
result = test_single_message(
    'pretrained_models/H3_MDP_StateMachine.pt',
    'Tell me more about this artwork',
    'Turban'
)
print(f"User: 'Tell me more about this artwork'")
print(f"Agent Response:")
print(f"  - Action: {result['action']}")
print(f"  - Option: {result['option']}")
print(f"  - Subaction: {result['subaction']}")
print(f"  - State Dimension: {result['state_vector'].shape[0]}")

# Test 3: Multi-turn conversation
print("\n" + "=" * 70)
print("TEST 3: Multi-turn Conversation - SMDP Model")
print("=" * 70)
messages = [
    ("King_Caspar", "Hello, what is this painting?"),
    ("King_Caspar", "Tell me more about the artist"),
    ("Turban", "What about this one?"),
]

results = test_conversation(
    'pretrained_models/H3_SMDP_StateMachine.pt',
    messages,
    starting_exhibit="King_Caspar"
)

for i, (exhibit, user_msg) in enumerate(messages):
    result = results[i]
    print(f"\nTurn {i+1} [{exhibit}]:")
    print(f"  User: '{user_msg}'")
    print(f"  Agent: {result['action']} ({result['option']}/{result['subaction']})")

# Test 4: Compare MDP vs SMDP on same input
print("\n" + "=" * 70)
print("TEST 4: Compare MDP vs SMDP on Same Input")
print("=" * 70)
user_message = "What can you tell me about this painting?"
exhibit = "King_Caspar"

mdp_result = test_single_message(
    'pretrained_models/H3_MDP_StateMachine.pt',
    user_message,
    exhibit
)

smdp_result = test_single_message(
    'pretrained_models/H3_SMDP_StateMachine.pt',
    user_message,
    exhibit
)

print(f"User: '{user_message}'")
print(f"\nMDP Model:  {mdp_result['action']}")
print(f"SMDP Model: {smdp_result['action']}")

# Test 5: Load model and inspect metadata
print("\n" + "=" * 70)
print("TEST 5: Model Metadata Inspection")
print("=" * 70)
model_data = load_trained_model('pretrained_models/H3_SMDP_StateMachine.pt')
print(f"Model Type: {model_data['model_type']}")
print(f"State Dimension: {model_data['state_dim']}")
print(f"Options: {model_data['options']}")
print(f"Subactions:")
for opt, subs in model_data['subactions'].items():
    print(f"  - {opt}: {subs}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE!")
print("=" * 70)
