# Model Inference Testing

Simple Python functions for testing trained HRL museum dialogue agent models.

## Quick Start

```python
from test_model import test_single_message

# Test a model with a single message
result = test_single_message(
    model_path="../pretrained_models/H3_SMDP_StateMachine.pt",
    user_message="What is this painting?",
    exhibit="King_Caspar"
)

print(result['action'])  # e.g., "Explain/ExplainNewFact"
```

## Installation

This inference module uses the same dependencies as the main training code. Install from the parent directory:

```bash
cd ..
pip install -r requirements.txt
```

## Available Models

Models are located in `../pretrained_models/`:

- `H3_MDP_StateMachine.pt` - Flat MDP model (baseline)
- `H3_SMDP_StateMachine.pt` - Hierarchical SMDP model (Option-Critic)

## Function Reference

### `load_trained_model(model_path, knowledge_graph_path=None, device='cpu')`

Load a trained model checkpoint and return agent, metadata, and helpers.

**Parameters:**
- `model_path` (str): Path to model checkpoint (.pt file)
- `knowledge_graph_path` (str, optional): Path to knowledge graph JSON (default: `../museum_knowledge_graph.json`)
- `device` (str): Device to load model on ('cpu' or 'cuda')

**Returns:**
Dictionary with:
- `agent`: ActorCriticAgent or FlatActorCriticAgent instance
- `options`: List of option names
- `subactions`: Dict mapping options to subaction lists
- `state_dim`: State dimension
- `knowledge_graph`: SimpleKnowledgeGraph instance
- `model_type`: 'MDP' or 'SMDP'
- `projection_matrix`: DialogueBERT projection matrix
- `bert_recognizer`: DialogueBERT recognizer instance

**Example:**
```python
model_data = load_trained_model("../pretrained_models/H3_SMDP_StateMachine.pt")
print(f"Model type: {model_data['model_type']}")
print(f"Options: {model_data['options']}")
```

### `test_single_message(model_path, user_message, exhibit="King_Caspar", dialogue_history=None, knowledge_graph_path=None, device='cpu')`

Test model with a single message.

**Parameters:**
- `model_path` (str): Path to model checkpoint
- `user_message` (str): User's message to test
- `exhibit` (str): Exhibit name (default: "King_Caspar")
- `dialogue_history` (list, optional): List of (role, utterance, turn_number) tuples
- `knowledge_graph_path` (str, optional): Path to knowledge graph
- `device` (str): Device to use ('cpu' or 'cuda')

**Returns:**
Dictionary with:
- `action`: Selected action string (e.g., "Explain/ExplainNewFact")
- `option`: Selected option name
- `subaction`: Selected subaction name
- `state_vector`: State vector used for inference
- `action_dict`: Full action dictionary from agent

**Example:**
```python
result = test_single_message(
    model_path="../pretrained_models/H3_SMDP_StateMachine.pt",
    user_message="What is this painting?",
    exhibit="King_Caspar"
)
print(f"Agent selected: {result['action']}")
```

### `test_conversation(model_path, messages, starting_exhibit="King_Caspar", knowledge_graph_path=None, device='cpu')`

Test model with multi-turn conversation.

**Parameters:**
- `model_path` (str): Path to model checkpoint
- `messages` (list): List of (exhibit, user_message) tuples
- `starting_exhibit` (str): Starting exhibit name
- `knowledge_graph_path` (str, optional): Path to knowledge graph
- `device` (str): Device to use ('cpu' or 'cuda')

**Returns:**
List of result dictionaries (one per message)

**Example:**
```python
messages = [
    ("King_Caspar", "Hello, what is this painting?"),
    ("King_Caspar", "Tell me more about the artist"),
    ("Turban", "What about this one?"),
]

results = test_conversation(
    model_path="../pretrained_models/H3_SMDP_StateMachine.pt",
    messages=messages
)

for i, result in enumerate(results):
    print(f"Turn {i+1}: {result['action']}")
```

### `get_agent_response(agent, user_message, exhibit, dialogue_history, knowledge_graph, options, subactions, facts_mentioned, option_counts, turn_number, projection_matrix, bert_recognizer)`

Core inference function - get agent's action selection.

**Parameters:**
- `agent`: Loaded agent instance
- `user_message`: User's current utterance
- `exhibit`: Current exhibit name
- `dialogue_history`: List of (role, utterance, turn_number) tuples
- `knowledge_graph`: Knowledge graph instance
- `options`: List of option names
- `subactions`: Dict mapping options to subaction lists
- `facts_mentioned`: Dict mapping exhibit names to sets of mentioned fact IDs
- `option_counts`: Dict mapping option names to usage counts
- `turn_number`: Current turn number
- `projection_matrix`: DialogueBERT projection matrix
- `bert_recognizer`: DialogueBERT recognizer instance

**Returns:**
Dictionary with action selection and state vector

**Example:**
```python
model_data = load_trained_model("../pretrained_models/H3_SMDP_StateMachine.pt")

result = get_agent_response(
    agent=model_data['agent'],
    user_message="What is this?",
    exhibit="King_Caspar",
    dialogue_history=[],
    knowledge_graph=model_data['knowledge_graph'],
    options=model_data['options'],
    subactions=model_data['subactions'],
    facts_mentioned=defaultdict(set),
    option_counts=defaultdict(int),
    turn_number=0,
    projection_matrix=model_data['projection_matrix'],
    bert_recognizer=model_data['bert_recognizer']
)
```

## Usage Examples

See `example_usage.py` for comprehensive examples:

- Load and test a model
- Single message test
- Multi-turn conversation
- Compare MDP vs SMDP models
- Extract state vectors for analysis
- Advanced usage with custom state

Run examples:
```bash
python example_usage.py
```

## State Vector Structure

The state vector has the following components (matching `env.py` `_get_obs()`):

1. **Focus vector**: `(n_exhibits + 1)`-d one-hot encoding of current exhibit
2. **History vector**: `(n_exhibits + n_options)`-d (exhibit coverage ratios + normalized option usage)
3. **Intent embedding**: `64`-d (DialogueBERT 768→64 projection)
4. **Context embedding**: `64`-d (DialogueBERT 768→64 projection)
5. **Subaction availability**: `4`-d binary indicators

**Total dimension**: `(n_exhibits + 1) + (n_exhibits + n_options) + 64 + 64 + 4`

For 5 exhibits: `(5+1) + (5+4) + 64 + 64 + 4 = 147`-d

## Available Exhibits

- `King_Caspar` - Painting of one of the three magi
- `Turban` - Portrait of a boy in oriental attire
- `Dom_Miguel` - Colonial-era portrait
- `Pedro_Sunda` - Historical figure portrait
- `Diego_Bemba` - Historical figure portrait

## Available Options and Subactions

**Options:**
- `Explain` - Explain facts about exhibits
- `AskQuestion` - Ask questions to engage visitor
- `OfferTransition` - Suggest moving to another exhibit
- `Conclude` - Wrap up the conversation

**Subactions:**
- `Explain`: `ExplainNewFact`, `RepeatFact`, `ClarifyFact`
- `AskQuestion`: `AskOpinion`, `AskMemory`, `AskClarification`
- `OfferTransition`: `SummarizeAndSuggest`
- `Conclude`: `WrapUp`

## Troubleshooting

### "Model checkpoint not found"
- Check that model path is correct
- Use relative path from `inference/` directory: `../pretrained_models/H3_SMDP_StateMachine.pt`
- Or use absolute path

### "Knowledge graph not found"
- Default path is `../museum_knowledge_graph.json`
- Ensure file exists in `training_export/` directory
- Or provide custom path via `knowledge_graph_path` parameter

### "State dimension mismatch"
- State dimension must match checkpoint `state_dim` exactly
- Check that number of exhibits matches training configuration
- Verify projection matrix uses same seed (42) as training

### "CUDA out of memory"
- Use `device='cpu'` instead of `device='cuda'`
- Inference is fast on CPU for single messages

## Notes

- **No LLM required**: Functions return action selections only (no dialogue generation)
- **Deterministic inference**: Always uses `deterministic=True` for consistent results
- **State consistency**: State construction matches `env.py` exactly
- **Action masking**: Automatically applied based on exhibit state and facts mentioned
- **DialogueBERT caching**: Model loads on first use and is cached

## File Structure

```
inference/
├── test_model.py          # Main inference functions
├── model_loader.py        # Model loading utilities
├── state_builder.py       # State vector construction
├── example_usage.py       # Usage examples
└── README.md             # This file
```

## Dependencies

- `torch` - PyTorch for model loading
- `numpy` - Numerical operations
- `src/agent/` - Agent implementations
- `src/utils/` - DialogueBERT, knowledge graph utilities

All dependencies are included in `../requirements.txt`.
