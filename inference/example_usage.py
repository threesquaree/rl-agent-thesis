"""
Example Usage of Model Inference Functions

Demonstrates how to use the inference functions to test trained models.
"""

from test_model import (
    load_trained_model,
    test_single_message,
    test_conversation,
    get_agent_response
)
from collections import defaultdict


def example_load_and_test():
    """Example: Load model and test with single message."""
    print("=" * 60)
    print("Example 1: Load Model and Test Single Message")
    print("=" * 60)
    
    # Load model
    model_path = "../pretrained_models/H3_SMDP_StateMachine.pt"
    model_data = load_trained_model(model_path)
    
    print(f"Loaded {model_data['model_type']} model")
    print(f"Options: {model_data['options']}")
    print(f"State dimension: {model_data['state_dim']}")
    
    # Test single message
    result = test_single_message(
        model_path=model_path,
        user_message="What is this painting?",
        exhibit="King_Caspar"
    )
    
    print(f"\nUser: 'What is this painting?'")
    print(f"Agent selected: {result['action']}")
    print(f"  Option: {result['option']}")
    print(f"  Subaction: {result['subaction']}")
    print(f"State vector shape: {result['state_vector'].shape}")


def example_single_message():
    """Example: Test with single message using convenience function."""
    print("\n" + "=" * 60)
    print("Example 2: Single Message Test")
    print("=" * 60)
    
    result = test_single_message(
        model_path="../pretrained_models/H3_MDP_StateMachine.pt",
        user_message="Tell me more about this artwork",
        exhibit="Turban"
    )
    
    print(f"Action: {result['action']}")
    print(f"State dimension: {result['state_vector'].shape[0]}")


def example_multi_turn_conversation():
    """Example: Test with multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)
    
    messages = [
        ("King_Caspar", "Hello, what is this painting?"),
        ("King_Caspar", "Tell me more about the artist"),
        ("Turban", "What about this one?"),
    ]
    
    results = test_conversation(
        model_path="../pretrained_models/H3_SMDP_StateMachine.pt",
        messages=messages,
        starting_exhibit="King_Caspar"
    )
    
    for i, (exhibit, user_msg) in enumerate(messages):
        result = results[i]
        print(f"\nTurn {i+1} [{exhibit}]:")
        print(f"  User: '{user_msg}'")
        print(f"  Agent: {result['action']}")


def example_compare_models():
    """Example: Compare MDP vs SMDP models."""
    print("\n" + "=" * 60)
    print("Example 4: Compare MDP vs SMDP Models")
    print("=" * 60)
    
    user_message = "What can you tell me about this painting?"
    exhibit = "King_Caspar"
    
    # Test MDP model
    mdp_result = test_single_message(
        model_path="../pretrained_models/H3_MDP_StateMachine.pt",
        user_message=user_message,
        exhibit=exhibit
    )
    
    # Test SMDP model
    smdp_result = test_single_message(
        model_path="../pretrained_models/H3_SMDP_StateMachine.pt",
        user_message=user_message,
        exhibit=exhibit
    )
    
    print(f"User: '{user_message}'")
    print(f"\nMDP Model:  {mdp_result['action']}")
    print(f"SMDP Model: {smdp_result['action']}")


def example_extract_state_vector():
    """Example: Extract and inspect state vector."""
    print("\n" + "=" * 60)
    print("Example 5: Extract State Vector for Analysis")
    print("=" * 60)
    
    model_data = load_trained_model("../pretrained_models/H3_SMDP_StateMachine.pt")
    
    # Build state manually
    from state_builder import build_state
    
    dialogue_history = [
        ("agent", "Welcome to the museum!", 0),
        ("user", "Hello", 0)
    ]
    
    state = build_state(
        user_message="What is this?",
        exhibit="King_Caspar",
        dialogue_history=dialogue_history,
        knowledge_graph=model_data['knowledge_graph'],
        options=model_data['options'],
        facts_mentioned=defaultdict(set),
        option_counts=defaultdict(int),
        turn_number=1,
        projection_matrix=model_data['projection_matrix'],
        bert_recognizer=model_data['bert_recognizer']
    )
    
    print(f"State vector shape: {state.shape}")
    print(f"State vector dtype: {state.dtype}")
    print(f"State vector range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"\nFirst 10 elements: {state[:10]}")


def example_advanced_usage():
    """Example: Advanced usage with custom dialogue history."""
    print("\n" + "=" * 60)
    print("Example 6: Advanced Usage with Custom State")
    print("=" * 60)
    
    model_data = load_trained_model("../pretrained_models/H3_SMDP_StateMachine.pt")
    
    # Custom dialogue history
    dialogue_history = [
        ("agent", "Welcome! I'm your museum guide.", 0),
        ("user", "Hi, I'm interested in the paintings.", 0),
        ("agent", "Great! Let's start with this one.", 1),
        ("user", "What is it?", 1)
    ]
    
    # Custom facts mentioned (simulating previous conversation)
    facts_mentioned = defaultdict(set)
    facts_mentioned["King_Caspar"].add("KC_001")  # Simulate one fact mentioned
    
    # Custom option counts
    option_counts = defaultdict(int)
    option_counts["Explain"] = 2
    option_counts["AskQuestion"] = 1
    
    result = get_agent_response(
        agent=model_data['agent'],
        user_message="Tell me more",
        exhibit="King_Caspar",
        dialogue_history=dialogue_history,
        knowledge_graph=model_data['knowledge_graph'],
        options=model_data['options'],
        subactions=model_data['subactions'],
        facts_mentioned=facts_mentioned,
        option_counts=option_counts,
        turn_number=2,
        projection_matrix=model_data['projection_matrix'],
        bert_recognizer=model_data['bert_recognizer']
    )
    
    print(f"User: 'Tell me more'")
    print(f"Agent selected: {result['action']}")
    print(f"State vector shape: {result['state_vector'].shape}")


if __name__ == "__main__":
    # Run examples
    try:
        example_load_and_test()
        example_single_message()
        example_multi_turn_conversation()
        example_compare_models()
        example_extract_state_vector()
        example_advanced_usage()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure model files exist in ../pretrained_models/")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
