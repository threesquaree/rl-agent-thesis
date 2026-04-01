# Simulator package for HRL Museum Dialogue Agent

"""
Simulator Factory

Provides a factory function to create simulators based on type:
- sim8: Original Sim8Simulator with persona behavior and gaze synthesis
- state_machine: New StateMachineSimulator with literature-grounded state machine
- hybrid: HybridSimulator combining state machine backbone + sim8 engagement dynamics

Usage:
    from src.simulator import get_simulator

    # Original simulator
    simulator = get_simulator("sim8", knowledge_graph=kg)

    # State machine simulator
    simulator = get_simulator("state_machine", knowledge_graph=kg)

    # Hybrid simulator (stochasticity controls sim8 influence: 0.0-1.0)
    simulator = get_simulator("hybrid", knowledge_graph=kg, stochasticity=0.5)
"""

from typing import Optional, List


def get_simulator(
    simulator_type: str = "sim8",
    knowledge_graph=None,
    exhibits: Optional[List[str]] = None,
    seed: int = 42,
    **kwargs
):
    """
    Factory function to create simulators.
    
    Args:
        simulator_type: "sim8" (original) or "state_machine" (new literature-grounded)
        knowledge_graph: SimpleKnowledgeGraph instance
        exhibits: List of exhibit names (fallback if no knowledge_graph)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to simulator
    
    Returns:
        Simulator instance (Sim8Simulator or StateMachineSimulator)
    
    Raises:
        ValueError: If unknown simulator_type
    """
    if simulator_type == "sim8":
        from .sim8_adapter import Sim8Simulator
        return Sim8Simulator(
            knowledge_graph=knowledge_graph,
            exhibits=exhibits,
            seed=seed,
            **kwargs
        )
    elif simulator_type == "state_machine":
        from .state_machine_simulator import StateMachineSimulator
        return StateMachineSimulator(
            knowledge_graph=knowledge_graph,
            exhibits=exhibits,
            seed=seed,
            **kwargs
        )
    elif simulator_type == "hybrid":
        from .hybrid_simulator import HybridSimulator
        stochasticity = kwargs.pop("stochasticity", 0.5)
        return HybridSimulator(
            knowledge_graph=knowledge_graph,
            exhibits=exhibits,
            seed=seed,
            stochasticity=stochasticity,
            **kwargs
        )
    elif simulator_type == "sim8_original":
        from .sim8_original_adapter import Sim8OriginalSimulator
        from pathlib import Path
        
        # Resolve model paths relative to project root (models/sim8_original/)
        # Path: src/simulator/__init__.py -> go up to project root
        project_root = Path(__file__).parent.parent.parent  # src/simulator -> src -> project_root
        dialogue_path = project_root / "models" / "sim8_original" / "visitor_dialogue"
        gaze_path = project_root / "models" / "sim8_original" / "gaze_vae_with_parent.pth"
        
        return Sim8OriginalSimulator(
            knowledge_graph=knowledge_graph,
            exhibits=exhibits,
            dialogue_model_path=str(dialogue_path),
            gaze_model_path=str(gaze_path),
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown simulator type: '{simulator_type}'. "
            f"Valid options: 'sim8', 'sim8_original', 'state_machine', 'hybrid'"
        )


# Convenience imports
from .sim8_adapter import Sim8Simulator

__all__ = ["get_simulator", "Sim8Simulator"]