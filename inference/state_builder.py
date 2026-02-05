"""
State Builder for Inference

Constructs state vectors for model inference (simplified version of env.py _get_obs()).
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dialoguebert_intent_recognizer import get_dialoguebert_recognizer
from src.utils.knowledge_graph import SimpleKnowledgeGraph


def get_projection_matrix() -> np.ndarray:
    """
    Get the projection matrix for DialogueBERT embeddings (768-d -> 64-d).
    
    Uses same seed (42) as training to ensure consistency.
    
    Returns:
        Projection matrix of shape (64, 768)
    """
    np.random.seed(42)  # Fixed seed for reproducibility (matches env.py line 167)
    return np.random.randn(64, 768).astype(np.float32) / np.sqrt(768)


def _build_focus_vector(exhibit_idx: int, n_exhibits: int) -> np.ndarray:
    """
    Build focus vector (one-hot encoding of current exhibit).
    
    Args:
        exhibit_idx: Index of current exhibit (0-indexed, or -1 for no focus)
        n_exhibits: Total number of exhibits
        
    Returns:
        Focus vector of shape (n_exhibits + 1,)
    """
    focus = np.zeros(n_exhibits + 1, dtype=np.float32)
    if 0 <= exhibit_idx < n_exhibits:
        focus[exhibit_idx] = 1.0
    else:
        focus[-1] = 1.0  # No focus
    return focus


def _build_history_vector(
    facts_mentioned: Dict[str, set],
    knowledge_graph: SimpleKnowledgeGraph,
    option_counts: Dict[str, int],
    n_exhibits: int,
    options: List[str]
) -> np.ndarray:
    """
    Build history vector (exhibit coverage ratios + normalized option usage).
    
    Args:
        facts_mentioned: Dict mapping exhibit names to sets of mentioned fact IDs
        knowledge_graph: Knowledge graph instance
        option_counts: Dict mapping option names to usage counts
        n_exhibits: Total number of exhibits
        options: List of option names
        
    Returns:
        History vector of shape (n_exhibits + len(options),)
    """
    history = np.zeros(n_exhibits + len(options), dtype=np.float32)
    exhibit_names = knowledge_graph.get_exhibit_names()
    
    # Exhibit completion ratios (first n_exhibits elements)
    for i, exhibit_name in enumerate(exhibit_names):
        all_facts = knowledge_graph.get_exhibit_facts(exhibit_name)
        total_facts = len(all_facts)
        mentioned_count = len(facts_mentioned.get(exhibit_name, set()))
        coverage = mentioned_count / total_facts if total_facts > 0 else 0.0
        history[i] = coverage
    
    # Option usage counts (normalized) - next len(options) elements
    total_actions = sum(option_counts.values()) or 1
    for i, opt in enumerate(options):
        history[n_exhibits + i] = option_counts.get(opt, 0) / total_actions
    
    return history


def _build_intent_embedding(
    user_message: str,
    turn_number: int,
    bert_recognizer,
    projection_matrix: np.ndarray
) -> np.ndarray:
    """
    Build intent embedding from user message using DialogueBERT.
    
    Args:
        user_message: User's utterance
        turn_number: Current turn number
        bert_recognizer: DialogueBERT recognizer instance
        projection_matrix: Projection matrix (64, 768)
        
    Returns:
        Projected intent embedding of shape (64,)
    """
    intent_768 = bert_recognizer.get_intent_embedding(
        user_message, role="user", turn_number=turn_number
    )
    intent_64 = np.dot(projection_matrix, intent_768).astype(np.float32)
    return intent_64


def _build_context_embedding(
    dialogue_history: List[Tuple[str, str, int]],
    bert_recognizer,
    projection_matrix: np.ndarray
) -> np.ndarray:
    """
    Build dialogue context embedding from recent history using DialogueBERT.
    
    Args:
        dialogue_history: List of (role, utterance, turn_number) tuples
        bert_recognizer: DialogueBERT recognizer instance
        projection_matrix: Projection matrix (64, 768)
        
    Returns:
        Projected context embedding of shape (64,)
    """
    context_768 = bert_recognizer.get_dialogue_context(dialogue_history, max_turns=3)
    context_64 = np.dot(projection_matrix, context_768).astype(np.float32)
    return context_64


def _build_subaction_availability(
    exhibit: str,
    knowledge_graph: SimpleKnowledgeGraph,
    facts_mentioned: Dict[str, set]
) -> np.ndarray:
    """
    Build subaction availability indicators.
    
    Args:
        exhibit: Current exhibit name
        knowledge_graph: Knowledge graph instance
        facts_mentioned: Dict mapping exhibit names to sets of mentioned fact IDs
        
    Returns:
        Availability vector of shape (4,):
        [0]: ExplainNewFact available (1.0) or masked (0.0)
        [1]: ClarifyFact available (1.0) or masked (0.0)
        [2]: RepeatFact available (1.0) or masked (0.0)
        [3]: Exhibit exhausted indicator (1.0 if exhausted, 0.0 otherwise)
    """
    availability = np.zeros(4, dtype=np.float32)
    
    if exhibit:
        all_facts = knowledge_graph.get_exhibit_facts(exhibit)
        mentioned_ids = facts_mentioned.get(exhibit, set())
        
        # Check if there are unmentioned facts
        has_unmentioned = any(
            knowledge_graph.extract_fact_id(f) not in mentioned_ids
            for f in all_facts
        )
        
        # [0]: ExplainNewFact available if there are unmentioned facts
        availability[0] = 1.0 if has_unmentioned else 0.0
        
        # [1]: ClarifyFact always available (no masking in original code)
        availability[1] = 1.0
        
        # [2]: RepeatFact available if at least one fact mentioned
        availability[2] = 1.0 if len(mentioned_ids) > 0 else 0.0
        
        # [3]: Exhibit exhausted if all facts mentioned
        availability[3] = 1.0 if not has_unmentioned and len(all_facts) > 0 else 0.0
    
    return availability


def build_state(
    user_message: str,
    exhibit: str,
    dialogue_history: List[Tuple[str, str, int]],
    knowledge_graph: SimpleKnowledgeGraph,
    options: List[str],
    facts_mentioned: Dict[str, set],
    option_counts: Dict[str, int],
    turn_number: int,
    projection_matrix: Optional[np.ndarray] = None,
    bert_recognizer = None,
    include_availability: bool = False
) -> np.ndarray:
    """
    Build complete state vector for inference.
    
    State components (matching env.py _get_obs()):
    1. Focus vector: (n_exhibits + 1)-d one-hot
    2. History vector: (n_exhibits + n_options)-d (coverage + usage)
    3. Intent embedding: 64-d (DialogueBERT 768→64 projection)
    4. Context embedding: 64-d (DialogueBERT 768→64 projection)
    5. Subaction availability: 4-d binary indicators
    
    Args:
        user_message: User's current utterance
        exhibit: Current exhibit name
        dialogue_history: List of (role, utterance, turn_number) tuples
        knowledge_graph: Knowledge graph instance
        options: List of option names
        facts_mentioned: Dict mapping exhibit names to sets of mentioned fact IDs
        option_counts: Dict mapping option names to usage counts
        turn_number: Current turn number
        projection_matrix: Optional pre-computed projection matrix
        bert_recognizer: Optional pre-initialized DialogueBERT recognizer
        
    Returns:
        Complete state vector as numpy array
    """
    # Initialize helpers if not provided
    if projection_matrix is None:
        projection_matrix = get_projection_matrix()
    
    if bert_recognizer is None:
        bert_recognizer = get_dialoguebert_recognizer()
    
    # Get exhibit information
    exhibit_names = knowledge_graph.get_exhibit_names()
    n_exhibits = len(exhibit_names)
    
    # Get exhibit index (0-indexed)
    if exhibit in exhibit_names:
        exhibit_idx = exhibit_names.index(exhibit)
    else:
        exhibit_idx = -1  # No focus
    
    # Build each component
    focus = _build_focus_vector(exhibit_idx, n_exhibits)
    history = _build_history_vector(facts_mentioned, knowledge_graph, option_counts, n_exhibits, options)
    intent_64 = _build_intent_embedding(user_message, turn_number, bert_recognizer, projection_matrix)
    context_64 = _build_context_embedding(dialogue_history, bert_recognizer, projection_matrix)
    
    # Concatenate base components
    state_components = [focus, history, intent_64, context_64]
    
    # Add availability indicators only if requested (for newer models)
    if include_availability:
        availability = _build_subaction_availability(exhibit, knowledge_graph, facts_mentioned)
        state_components.append(availability)
    
    # Concatenate into final state vector
    state = np.concatenate(state_components).astype(np.float32)
    
    return state
