"""
Model Loader for Inference

Loads trained model checkpoints and initializes appropriate agents (MDP or SMDP).
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.actor_critic_agent import ActorCriticAgent
from src.flat_rl.agent import FlatActorCriticAgent


def detect_model_type(checkpoint: dict) -> str:
    """
    Detect if model is flat (MDP) or hierarchical (SMDP).
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        'MDP' or 'SMDP'
    """
    # Check network keys to determine type
    if 'agent_state_dict' in checkpoint:
        network_keys = checkpoint['agent_state_dict'].keys()
    elif 'network' in checkpoint:
        network_keys = checkpoint['network'].keys()
    else:
        # Default to MDP if can't determine
        return 'MDP'
    
    # SMDP models have intra_option_policies and termination_functions
    for key in network_keys:
        if 'intra_option_policies' in key or 'termination_functions' in key:
            return 'SMDP'
    
    # MDP models have flat action logits
    return 'MDP'


def load_model_checkpoint(model_path: str, device: str = 'cpu') -> dict:
    """
    Load model checkpoint from file.
    
    Args:
        model_path: Path to .pt checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Loaded checkpoint dictionary
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    return checkpoint


def create_agent_from_checkpoint(
    checkpoint: dict,
    device: str = 'cpu',
    hidden_dim: int = 256,
    lstm_hidden_dim: int = 128,
    use_lstm: bool = True
) -> tuple:
    """
    Create agent instance from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        device: Device to create agent on
        hidden_dim: Hidden layer dimension (must match training)
        lstm_hidden_dim: LSTM hidden dimension (must match training)
        use_lstm: Whether to use LSTM (must match training)
        
    Returns:
        Tuple of (agent, model_type, metadata)
        - agent: ActorCriticAgent or FlatActorCriticAgent
        - model_type: 'MDP' or 'SMDP'
        - metadata: dict with state_dim, options, subactions
    """
    # Extract metadata
    state_dim = checkpoint.get('state_dim', 143)
    options = checkpoint.get('options', ["Explain", "AskQuestion", "OfferTransition", "Conclude"])
    subactions = checkpoint.get('subactions', {
        "Explain": ["ExplainNewFact", "RepeatFact", "ClarifyFact"],
        "AskQuestion": ["AskOpinion", "AskMemory", "AskClarification"],
        "OfferTransition": ["SummarizeAndSuggest"],
        "Conclude": ["WrapUp"]
    })
    
    # Detect model type
    model_type = detect_model_type(checkpoint)
    
    # Create appropriate agent
    if model_type == 'SMDP':
        agent = ActorCriticAgent(
            state_dim=state_dim,
            options=options,
            subactions=subactions,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            use_lstm=use_lstm,
            device=device,
            termination_strategy='learned'
        )
    else:  # MDP
        agent = FlatActorCriticAgent(
            state_dim=state_dim,
            options=options,
            subactions=subactions,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            use_lstm=use_lstm,
            device=device
        )
    
    # Load weights
    if 'agent_state_dict' in checkpoint:
        agent.network.load_state_dict(checkpoint['agent_state_dict'])
    elif 'network' in checkpoint:
        agent.network.load_state_dict(checkpoint['network'])
    else:
        raise ValueError("Checkpoint does not contain 'agent_state_dict' or 'network' key")
    
    # Set to evaluation mode
    agent.network.eval()
    
    metadata = {
        'state_dim': state_dim,
        'options': options,
        'subactions': subactions,
        'model_type': model_type
    }
    
    return agent, model_type, metadata
