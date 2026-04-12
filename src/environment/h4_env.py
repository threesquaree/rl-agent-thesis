"""
H4 Environment Variant: State Ablation (Dialogue Act Representation)

Extends base MuseumDialogueEnv to use dialogue-act state representation
instead of full DialogueBERT embeddings for hypothesis testing.

State: [f_t, h_t, a_t] where:
- f_t: focus vector (n_exhibits + 1)
- h_t: dialogue history (n_exhibits + n_subactions)
- a_t: dialogue act probability distribution (8-d)
Total: ~23-d for 5 exhibits (vs 149-d with DialogueBERT)

Uses HuggingFace zero-shot classification with soft probabilities (not one-hot)
to preserve model uncertainty and provide richer semantic understanding.
"""

from src.environment.env import MuseumDialogueEnv
from src.environment.dialogue_act_classifier import get_dialogue_act_classifier
import numpy as np


class H5StateAblationEnv(MuseumDialogueEnv):
    """
    Environment variant for H4 hypothesis: state ablation using dialogue acts.
    
    Replaces DialogueBERT embeddings (i_t, c_t) with compact dialogue act
    classification (a_t) to test if reduced state dimension maintains performance.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize H4 environment variant."""
        # Initialize dialogue act classifier BEFORE super().__init__()
        # because super().__init__() calls reset() which calls _get_obs()
        self.act_classifier = get_dialogue_act_classifier()
        self.last_dialogue_act = "statement"  # Track previous act for context
        
        super().__init__(*args, **kwargs)
        
        # Override observation space dimension
        # Focus: n_exhibits + 1
        # History: n_exhibits + n_subactions (per-subaction usage)
        # Dialogue act: 8 (probability distribution)
        act_dim = self.act_classifier.get_state_dim()  # 8-d
        new_obs_dim = (self.n_exhibits + 1) + (self.n_exhibits + len(self._all_subactions)) + act_dim
        
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
        
        print(f"[H4] State ablation: {new_obs_dim}-d (vs {149}-d baseline)")
        print(f"[H4] Dialogue act representation: 8-d probability distribution (soft labels)")
    
    def _get_obs(self):
        """
        Construct observation with dialogue-act state (H4 variant).
        
        State: [f_t, h_t, a_t]
        - f_t: focus vector (n_exhibits + 1)
        - h_t: dialogue history (n_exhibits + n_subactions)
        - a_t: dialogue act probability distribution (8-d soft labels)
        """
        # 1. Focus vector f_t (same as baseline)
        focus_snapshot = np.zeros(self.n_exhibits + 1)
        if self.focus > 0:
            focus_snapshot[self.focus - 1] = 1.0
        else:
            focus_snapshot[-1] = 1.0
        
        # 2. Dialogue history h_t (same as baseline)
        history = np.zeros(self.n_exhibits + len(self._all_subactions))
        coverage = self._get_museum_exhibit_coverage()

        for i, exhibit_name in enumerate(self.exhibit_keys):
            completion_ratio = coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
            history[i] = completion_ratio

        total_actions = sum(self.actions_used.values()) or 1
        for i, sa in enumerate(self._all_subactions):
            history[self.n_exhibits + i] = self.actions_used[sa] / total_actions
        
        # 3. Dialogue act a_t (H4: replaces DialogueBERT embeddings)
        # Use zero-shot classification with soft probabilities
        # This preserves model uncertainty and provides richer semantic understanding
        act_result = self.act_classifier.classify_with_probabilities(
            self.last_user_utterance,
            previous_act=self.last_dialogue_act
        )
        act_vector = act_result['probabilities']  # 8-d probability distribution
        self.last_dialogue_act = act_result['label']  # Top label for tracking
        
        # Concatenate: [f_t, h_t, a_t]
        obs = np.concatenate([
            focus_snapshot,  # (n_exhibits + 1)-d
            history,        # (n_exhibits + 4)-d
            act_vector      # 8-d
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action_dict):
        """
        Override step to track dialogue acts in info.
        """
        obs, reward, done, truncated, info = super().step(action_dict)
        
        # Add dialogue act to info for evaluation
        info['dialogue_act'] = self.last_dialogue_act
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and dialogue act state."""
        obs, info = super().reset(seed=seed, options=options)
        self.last_dialogue_act = "statement"  # Reset dialogue act
        return obs, info

