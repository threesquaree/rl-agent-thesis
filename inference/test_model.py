"""
Model Inference Testing Script

Simple Python script with functions for testing trained models.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

# Add parent directory to path to import src modules
inference_dir = Path(__file__).parent
sys.path.insert(0, str(inference_dir.parent))
sys.path.insert(0, str(inference_dir))

from model_loader import load_model_checkpoint, create_agent_from_checkpoint
from state_builder import build_state, get_projection_matrix
from src.utils.knowledge_graph import SimpleKnowledgeGraph
from src.utils.dialoguebert_intent_recognizer import get_dialoguebert_recognizer


def load_trained_model(
    model_path: str,
    knowledge_graph_path: Optional[str] = None,
    device: str = 'cpu'
) -> dict:
    """
    Load trained model and return agent, metadata, and helpers.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        knowledge_graph_path: Path to knowledge graph JSON (default: ../museum_knowledge_graph.json)
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with:
        - 'agent': ActorCriticAgent or FlatActorCriticAgent
        - 'options': List of option names
        - 'subactions': Dict mapping options to subaction lists
        - 'state_dim': State dimension
        - 'knowledge_graph': SimpleKnowledgeGraph instance
        - 'model_type': 'MDP' or 'SMDP'
        - 'projection_matrix': DialogueBERT projection matrix
        - 'bert_recognizer': DialogueBERT recognizer instance
    """
    # Resolve paths
    inference_dir = Path(__file__).parent
    training_export_dir = inference_dir.parent
    
    if not Path(model_path).is_absolute():
        model_path = training_export_dir / model_path
    
    # Load knowledge graph
    if knowledge_graph_path is None:
        knowledge_graph_path = training_export_dir / "museum_knowledge_graph.json"
    else:
        knowledge_graph_path = Path(knowledge_graph_path)
    
    if not knowledge_graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {knowledge_graph_path}")
    
    knowledge_graph = SimpleKnowledgeGraph(str(knowledge_graph_path))
    
    # Load checkpoint
    checkpoint = load_model_checkpoint(str(model_path), device=device)
    
    # Create agent
    agent, model_type, metadata = create_agent_from_checkpoint(checkpoint, device=device)
    
    # Initialize helpers
    projection_matrix = get_projection_matrix()
    bert_recognizer = get_dialoguebert_recognizer()
    
    return {
        'agent': agent,
        'options': metadata['options'],
        'subactions': metadata['subactions'],
        'state_dim': metadata['state_dim'],
        'knowledge_graph': knowledge_graph,
        'model_type': model_type,
        'projection_matrix': projection_matrix,
        'bert_recognizer': bert_recognizer
    }


def get_available_actions(
    exhibit: str,
    knowledge_graph: SimpleKnowledgeGraph,
    facts_mentioned: Dict[str, set],
    options: List[str],
    subactions: Dict[str, List[str]]
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Get available options and subactions with action masking.
    
    Args:
        exhibit: Current exhibit name
        knowledge_graph: Knowledge graph instance
        facts_mentioned: Dict mapping exhibit names to sets of mentioned fact IDs
        options: List of all option names
        subactions: Dict mapping options to subaction lists
        
    Returns:
        Tuple of (available_options, available_subactions_dict)
    """
    available_options = list(options)
    available_subactions_dict = {}
    
    for option in options:
        available_subs = list(subactions.get(option, []))
        
        # Apply masking for Explain option
        if option == "Explain" and exhibit:
            all_facts = knowledge_graph.get_exhibit_facts(exhibit)
            mentioned_ids = facts_mentioned.get(exhibit, set())
            
            # Check if there are unmentioned facts
            has_unmentioned = any(
                knowledge_graph.extract_fact_id(f) not in mentioned_ids
                for f in all_facts
            )
            
            # Remove ExplainNewFact if all facts mentioned
            if not has_unmentioned and "ExplainNewFact" in available_subs:
                available_subs.remove("ExplainNewFact")
            
            # Remove RepeatFact if no facts mentioned yet
            if len(mentioned_ids) == 0 and "RepeatFact" in available_subs:
                available_subs.remove("RepeatFact")
        
        if available_subs:  # Only include option if it has available subactions
            available_subactions_dict[option] = available_subs
        else:
            # Remove option if no subactions available
            if option in available_options:
                available_options.remove(option)
    
    return available_options, available_subactions_dict


def get_agent_response(
    agent,
    user_message: str,
    exhibit: str,
    dialogue_history: List[Tuple[str, str, int]],
    knowledge_graph: SimpleKnowledgeGraph,
    options: List[str],
    subactions: Dict[str, List[str]],
    facts_mentioned: Dict[str, set],
    option_counts: Dict[str, int],
    turn_number: int,
    projection_matrix: np.ndarray,
    bert_recognizer,
    state_dim: Optional[int] = None
) -> Dict[str, Any]:
    """
    Core inference function - get agent's action selection.
    
    Args:
        agent: Loaded agent instance
        user_message: User's current utterance
        exhibit: Current exhibit name
        dialogue_history: List of (role, utterance, turn_number) tuples
        knowledge_graph: Knowledge graph instance
        options: List of option names
        subactions: Dict mapping options to subaction lists
        facts_mentioned: Dict mapping exhibit names to sets of mentioned fact IDs
        option_counts: Dict mapping option names to usage counts
        turn_number: Current turn number
        projection_matrix: DialogueBERT projection matrix
        bert_recognizer: DialogueBERT recognizer instance
        
    Returns:
        Dictionary with:
        - 'action': Selected action string (e.g., "Explain/ExplainNewFact")
        - 'option': Selected option name
        - 'subaction': Selected subaction name
        - 'state_vector': State vector used for inference
        - 'action_dict': Full action dictionary from agent
    """
    # Determine if we should include availability indicators
    # Models trained with state_dim=143 don't have them, models with state_dim=147 do
    include_availability = False
    if state_dim is not None:
        # Calculate expected dimension without availability
        n_exhibits = len(knowledge_graph.get_exhibit_names())
        base_dim = (n_exhibits + 1) + (n_exhibits + len(options)) + 64 + 64
        include_availability = (state_dim == base_dim + 4)
    
    # Build state vector
    state_vector = build_state(
        user_message=user_message,
        exhibit=exhibit,
        dialogue_history=dialogue_history,
        knowledge_graph=knowledge_graph,
        options=options,
        facts_mentioned=facts_mentioned,
        option_counts=option_counts,
        turn_number=turn_number,
        projection_matrix=projection_matrix,
        bert_recognizer=bert_recognizer,
        include_availability=include_availability
    )
    
    # Get available actions with masking
    available_options, available_subactions_dict = get_available_actions(
        exhibit, knowledge_graph, facts_mentioned, options, subactions
    )
    
    # Get agent's action selection (deterministic for inference)
    action_dict = agent.select_action(
        state=state_vector,
        available_options=available_options,
        available_subactions_dict=available_subactions_dict,
        deterministic=True
    )
    
    option_name = action_dict.get('option_name', '')
    subaction_name = action_dict.get('subaction_name', '')
    action_string = f"{option_name}/{subaction_name}" if option_name and subaction_name else ""
    
    return {
        'action': action_string,
        'option': option_name,
        'subaction': subaction_name,
        'state_vector': state_vector,
        'action_dict': action_dict
    }


def test_single_message(
    model_path: str,
    user_message: str,
    exhibit: str = "King_Caspar",
    dialogue_history: Optional[List[Tuple[str, str, int]]] = None,
    knowledge_graph_path: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Test model with single message.
    
    Args:
        model_path: Path to model checkpoint
        user_message: User's message to test
        exhibit: Exhibit name (default: "King_Caspar")
        dialogue_history: Optional dialogue history (default: empty)
        knowledge_graph_path: Optional path to knowledge graph
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Dictionary with:
        - 'action': Selected action (e.g., "Explain/ExplainNewFact")
        - 'option': Selected option
        - 'subaction': Selected subaction
        - 'state_vector': State vector used
        - 'action_dict': Full action dictionary
    """
    # Load model
    model_data = load_trained_model(model_path, knowledge_graph_path, device)
    
    # Initialize conversation state
    if dialogue_history is None:
        dialogue_history = []
    
    facts_mentioned = defaultdict(set)
    option_counts = defaultdict(int)
    turn_number = len([e for e in dialogue_history if e[0] == "user"])
    
    # Get agent response
    result = get_agent_response(
        agent=model_data['agent'],
        user_message=user_message,
        exhibit=exhibit,
        dialogue_history=dialogue_history,
        knowledge_graph=model_data['knowledge_graph'],
        options=model_data['options'],
        subactions=model_data['subactions'],
        facts_mentioned=facts_mentioned,
        option_counts=option_counts,
        turn_number=turn_number,
        projection_matrix=model_data['projection_matrix'],
        bert_recognizer=model_data['bert_recognizer'],
        state_dim=model_data['state_dim']
    )
    
    return result


def test_conversation(
    model_path: str,
    messages: List[Tuple[str, str]],
    starting_exhibit: str = "King_Caspar",
    knowledge_graph_path: Optional[str] = None,
    device: str = 'cpu'
) -> List[Dict[str, Any]]:
    """
    Test model with multi-turn conversation.
    
    Args:
        model_path: Path to model checkpoint
        messages: List of (exhibit, user_message) tuples
        starting_exhibit: Starting exhibit name
        knowledge_graph_path: Optional path to knowledge graph
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        List of result dictionaries (one per message)
    """
    # Load model
    model_data = load_trained_model(model_path, knowledge_graph_path, device)
    
    # Initialize conversation state
    dialogue_history = []
    facts_mentioned = defaultdict(set)
    option_counts = defaultdict(int)
    current_exhibit = starting_exhibit
    
    results = []
    
    for exhibit, user_message in messages:
        # Update exhibit if provided
        if exhibit:
            current_exhibit = exhibit
        
        turn_number = len([e for e in dialogue_history if e[0] == "user"])
        
        # Get agent response
        result = get_agent_response(
            agent=model_data['agent'],
            user_message=user_message,
            exhibit=current_exhibit,
            dialogue_history=dialogue_history,
            knowledge_graph=model_data['knowledge_graph'],
            options=model_data['options'],
            subactions=model_data['subactions'],
            facts_mentioned=facts_mentioned,
            option_counts=option_counts,
            turn_number=turn_number,
            projection_matrix=model_data['projection_matrix'],
            bert_recognizer=model_data['bert_recognizer'],
            state_dim=model_data['state_dim']
        )
        
        results.append(result)
        
        # Update conversation state
        dialogue_history.append(("user", user_message, turn_number))
        option_counts[result['option']] += 1
        
        # Track mentioned facts (simplified - would need LLM response parsing in full version)
        # For now, just update turn number
        turn_number += 1
    
    return results
