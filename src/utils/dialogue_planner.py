"""
Dialogue Planner for HRL Museum Dialogue Agent

This module implements structured prompting for LLM-based dialogue generation in the
HRL museum agent. It translates high-level options and subactions into structured
prompts that guide LLM generation while maintaining dialogue coherence and policy
adherence.
"""

import re
from typing import List, Optional


def build_prompt(option: str, subaction: str, ex_id: Optional[str], 
                last_utt: str, facts_all: List[str], facts_used: List[str], 
                selected_fact: Optional[str], dialogue_history: List[str] = None,
                exhibit_names: List[str] = None, knowledge_graph=None,
                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """Build structured prompt for LLM dialogue generation"""
    
    # H6: Handle coarse option granularity by mapping coarse options to original option names
    # for prompt construction purposes
    if option == "Engage":
        # Map subaction to its original option for prompt routing
        from src.agent.option_configs import get_subaction_origin
        option = get_subaction_origin(subaction)
    elif option == "Transition":
        # H6 coarse_3opt: "Transition" option maps to "OfferTransition" for prompts
        option = "OfferTransition"
    
    # Build context - show facts ONLY for Explain option and Conclude option
    show_facts = (option == "Explain") or (option == "Conclude" and subaction == "SummarizeKeyPoints")
    context_section = _build_enhanced_context_section(ex_id, last_utt, facts_all, facts_used, dialogue_history, exhibit_names, knowledge_graph, show_facts=show_facts)

    # Calculate current exhibit completion for contextual prompts
    current_completion = 0.0
    if coverage_dict and ex_id:
        current_completion = coverage_dict.get(ex_id, {"coverage": 0.0})["coverage"]
    
    # Route to specific subaction function
    if option == "Explain":
        if subaction == "ExplainNewFact":
            return build_explain_new_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "RepeatFact":
            return build_repeat_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "ClarifyFact":
            return build_clarify_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)

    elif option == "AskQuestion":
        if subaction == "AskOpinion":
            return build_ask_opinion_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "AskMemory":
            return build_ask_memory_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "AskClarification":
            return build_ask_clarification_prompt(ex_id, context_section, facts_all, facts_used, current_completion)

    elif option == "OfferTransition":
        if subaction in ("SuggestMove", "SummarizeAndSuggest"):
            return build_offer_transition_prompt(ex_id, context_section, facts_all, facts_used, exhibit_names, knowledge_graph, target_exhibit, coverage_dict)

    elif option == "Conclude":
        if subaction == "WrapUp":
            return build_wrap_up_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "SummarizeKeyPoints":
            return build_summarize_key_points_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
    
    # Should never reach here - all options should be handled above
    raise ValueError(f"Unknown option '{option}' or subaction '{subaction}'")


def _build_enhanced_context_section(ex_id: Optional[str], last_utt: str, facts_all: List[str], 
                                  facts_used: List[str], dialogue_history: List = None,
                                  exhibit_names: List[str] = None, knowledge_graph=None, show_facts: bool = True) -> str:
    """Build enhanced context section with rich dialogue understanding
    
    Args:
        dialogue_history: List of (role, utterance) tuples where role is 'agent' or 'user'
    """
    import re
    context_parts = []
    
    # === EXHIBIT INFORMATION ===
    if ex_id:
        context_parts.append(f"CURRENT EXHIBIT: {ex_id.replace('_', ' ')}")
    
    # === VISITOR'S CURRENT MESSAGE ===
    if last_utt.strip():
        context_parts.append("VISITOR'S MESSAGE:")
        context_parts.append(f'"{last_utt}"')
        context_parts.append("")
    
    # === DIALOGUE HISTORY (for natural conversation flow) ===
    fact_ids_in_context = set()  # Track fact IDs already in conversation
    if dialogue_history and len(dialogue_history) > 0:
        # Get last 4 utterances (2 full exchanges: agent->user->agent->user)
        recent_context = dialogue_history[-4:] if len(dialogue_history) > 4 else dialogue_history
        context_parts.append("CONVERSATION CONTEXT (for natural flow):")
        for i, utterance_tuple in enumerate(recent_context, 1):
            # Handle both (role, utterance) and (role, utterance, turn_number) formats
            if len(utterance_tuple) >= 2:
                role, utterance = utterance_tuple[0], utterance_tuple[1]
            else:
                continue  # Skip invalid entries
            
            role_label = "AGENT" if role == "agent" else "VISITOR"
            context_parts.append(f'  {role_label}: "{utterance}"')
            
            # Extract fact IDs from this utterance
            fact_ids = re.findall(r'\[([A-Z]{2}_\d{3})\]', utterance)
            fact_ids_in_context.update(fact_ids)
        
        # CRITICAL: Warn about fact ID reuse
        if fact_ids_in_context:
            context_parts.append("")
            context_parts.append(f"FACTS ALREADY SHARED: {sorted(fact_ids_in_context)}")
            context_parts.append("(Use conversation context to build naturally on what was discussed)")
        context_parts.append("")

    # === SHOW FACTS ONLY IF ALLOWED (Explain or Summarize actions) ===
    # NOTE: For ExplainNewFact, facts_all is already filtered to unmentioned facts by env.py
    # The specific prompt builders (build_explain_new_fact_prompt) handle showing facts
    # So we DON'T show facts here to avoid duplication
    if not show_facts:
        # For Ask/Transition/Conclude - NO FACTS SHOWN
        context_parts.append("NO FACTS IN THIS RESPONSE TYPE")
        context_parts.append("Your job is to ask questions or suggest actions ONLY")
        context_parts.append("")
    
    return "\n".join(context_parts)


def _analyze_visitor_utterance(utterance: str) -> str:
    """Analyze visitor's utterance to understand their intent and interests"""
    utterance_lower = utterance.lower()
    
    # Question detection
    if any(word in utterance_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']) or '?' in utterance:
        if any(word in utterance_lower for word in ['more', 'tell me', 'explain', 'about']):
            return "Asking for more detailed information - wants deeper explanation"
        elif any(word in utterance_lower for word in ['meaning', 'significance', 'important']):
            return "Asking about meaning/significance - wants cultural/historical context"
        elif any(word in utterance_lower for word in ['made', 'created', 'built', 'constructed']):
            return "Asking about creation/construction - wants process/technique information"
        else:
            return "Asking a specific question - wants direct answer"
    
    # Interest/engagement detection
    elif any(word in utterance_lower for word in ['interesting', 'fascinating', 'amazing', 'beautiful', 'incredible']):
        return "Expressing positive interest - engaged and wants to learn more"
    
    # Confusion/clarification detection
    elif any(word in utterance_lower for word in ['confused', 'understand', 'unclear', 'not sure']):
        return "Expressing confusion - needs clarification or simpler explanation"
    
    # Agreement/acknowledgment
    elif any(word in utterance_lower for word in ['yes', 'ok', 'sure', 'i see', 'understand']):
        return "Acknowledging information - ready for next topic or deeper detail"
    
    # Personal connection
    elif any(word in utterance_lower for word in ['reminds me', 'similar', 'like', 'seen']):
        return "Making personal connections - engage with their experience"
    
    else:
        return "General engagement - continue educational dialogue"


# ===== EXPLAIN OPTION FUNCTIONS =====

def build_explain_new_fact_prompt(ex_id: Optional[str], context_section: str,
                                facts_all: List[str], facts_used: List[str],
                                selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for explaining a new fact about current exhibit"""
    
    # CRITICAL: Filter to only show NEW/unused facts
    used_ids = set()
    for fact in facts_used:
        # Extract fact ID from used facts
        import re
        match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact)
        if match:
            used_ids.add(match.group(1))
    
    # Filter out already-used facts
    new_facts = []
    for fact in facts_all:
        import re
        match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact)
        if match and match.group(1) not in used_ids:
            new_facts.append(fact)

    if not new_facts:
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ExplainNewFact policy for this turn:
1) First, directly answer the user's question in a natural way.
2) No new facts are available now. Do not force a new fact.
3) You lightly suggest moving to another exhibit.
4) Do not output any [FACT_ID].
5) Do not quote the visitor verbatim.
6) Maximum two sentences.

Response:"""

    new_facts_list = "\n".join([f"- {fact}" for fact in new_facts])
    used_ids_str = ", ".join(sorted(used_ids)) if used_ids else "None"

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ExplainNewFact policy for this turn:
1) Sentence 1: briefly answer the user's question.
2) Sentence 2: add one relevant new fact from the list below.
3) The second sentence must connect naturally to the first.
4) Include the exact [FACT_ID] for the fact you use.
5) Never repeat previously used IDs: {used_ids_str}.
6) Never invent IDs.
7) Never use facts outside the list below.
8) Do not quote the visitor verbatim.
9) Maximum two sentences.

New facts available (use only from this list):
{new_facts_list}

Response:"""


def build_repeat_fact_prompt(ex_id: Optional[str], context_section: str,
                           facts_all: List[str], facts_used: List[str],
                           selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for repeating a previously shared fact"""

    if facts_used:
        fact_to_repeat = selected_fact if selected_fact else facts_used[-1]

        # Extract fact ID from the fact string
        fact_id_match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact_to_repeat)
        fact_id = fact_id_match.group(1) if fact_id_match else ""
        fact_content = re.sub(r'\[([A-Z]{2}_\d{3})\]\s*', '', fact_to_repeat).strip()

        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. First answer the user's question naturally.
2. Then restate the selected previously-shared fact in fresh words.
3. Use the exact fact ID [{fact_id}] for that repeated fact.
4. Do not add any new facts or new IDs.
5. Do not quote the visitor verbatim.
6. Maximum two sentences.

Selected repeated fact:
- [{fact_id}] {fact_content}

Response:"""
    else:
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. No previously shared fact is available to repeat.
2. Answer the user's question directly without adding any [FACT_ID].
3. Do not invent fact IDs.
4. Maximum two sentences.

Response:"""


def build_clarify_fact_prompt(ex_id: Optional[str], context_section: str,
                            facts_all: List[str], facts_used: List[str],
                            selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for clarifying a fact"""
    if facts_used:
        fact_to_clarify = selected_fact if selected_fact else facts_used[-1]
        fact_plain = re.sub(r'^\s*\[[A-Z]{2}_\d{3}\]\s*', '', fact_to_clarify or '').strip()
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ClarifyFact policy:
1. First, respond naturally to the user's question or confusion.
2. Clarify this previously discussed fact in simpler words or with an everyday analogy:
   "{fact_plain}"
3. Do not introduce new facts.
4. Do not include any [FACT_ID].
5. Do not quote the visitor verbatim.
6. Maximum two sentences.

Response:"""

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ClarifyFact policy:
1. No previously discussed fact is available to clarify.
2. Answer the user's question directly in simple language.
3. Do not include any [FACT_ID].
4. Maximum two sentences.

Response:"""


# ===== ASK QUESTION OPTION FUNCTIONS =====

def build_ask_opinion_prompt(ex_id: Optional[str], context_section: str,
                               facts_all: List[str], facts_used: List[str],
                               current_completion: float = 0.0) -> str:
    """Build prompt for asking the visitor's opinion"""

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskOpinion policy:
1. First, respond naturally to the user's current input without quoting them verbatim.
2. Then ask exactly one opinion question.
3. The question must include at least one keyword from the current exhibit or current AOI.
4. The question must be opinion/feeling-oriented.
5. Do not introduce new facts.
6. Do not include any [FACT_ID].
7. Maximum two sentences.

Response:"""


def build_ask_memory_prompt(ex_id: Optional[str], context_section: str,
                          facts_all: List[str], facts_used: List[str],
                          current_completion: float = 0.0) -> str:
    """Build prompt for checking the visitor's memory"""

    if facts_used:
        recent_facts = facts_used[-3:]
        facts_block = "\n".join([f"- {f}" for f in recent_facts])
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskMemory policy:
1. First respond naturally to the user's current input.
2. Then ask one short memory-check question.
3. Memory target priority:
   - Highest priority: recall from facts already used in this exhibit.
   - If no usable fact, recall from recently mentioned points in conversation history.
4. Do not include [FACT_ID].
5. Maximum two sentences.

Recent used facts (priority memory source):
{facts_block}

Response:"""

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskMemory policy:
1. First respond naturally to the user's current input.
2. Then ask one short memory-check question.
3. Memory target: recently mentioned points in conversation history.
4. Do not include [FACT_ID].
5. Maximum two sentences.

Response:"""


def build_ask_clarification_prompt(ex_id: Optional[str], context_section: str,
                                 facts_all: List[str], facts_used: List[str],
                                 current_completion: float = 0.0) -> str:
    """Build prompt for asking for clarification"""
    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskClarification policy:
1. Follow this priority order: Answer > Summary > Clarify question.
2. Only ask a clarification question if ambiguity is genuine.
3. Do not include any [FACT_ID].
4. Do not quote the visitor verbatim.
5. Maximum two sentences.

Response:"""


# ===== OFFER TRANSITION OPTION FUNCTIONS =====

def _build_exhibit_inventory_section(exhibit_names: List[str], facts_used: List[str], knowledge_graph) -> str:
    """Build a section showing all exhibits and their exploration status"""
    if not exhibit_names or not knowledge_graph:
        return ""
    
    inventory_lines = ["MUSEUM EXHIBITS INVENTORY:", ""]
    
    # Convert facts_used to a set for faster lookup (plain text without IDs)
    facts_used_set = set(facts_used)
    
    # Calculate facts per exhibit
    exhibit_facts_count = {}
    exhibit_facts_used = {}
    
    for exhibit_name in exhibit_names:
        facts_all = knowledge_graph.get_exhibit_facts(exhibit_name) if knowledge_graph else []
        # Strip IDs from facts_all for comparison
        facts_remaining = [f for f in facts_all if knowledge_graph.strip_fact_id(f) not in facts_used_set]
        
        exhibit_facts_count[exhibit_name] = len(facts_all)
        exhibit_facts_used[exhibit_name] = len(facts_all) - len(facts_remaining)
    
    # Sort by unexplored facts (most first)
    sorted_exhibits = sorted(
        exhibit_names,
        key=lambda ex: (exhibit_facts_count.get(ex, 0) - exhibit_facts_used.get(ex, 0)),
        reverse=True
    )
    
    for exhibit_name in sorted_exhibits:
        total = exhibit_facts_count.get(exhibit_name, 0)
        used = exhibit_facts_used.get(exhibit_name, 0)
        remaining = total - used
        
        status_icon = "✓" if used == total else "◐" if used > 0 else "○"
        inventory_lines.append(
            f"  {status_icon} {exhibit_name.replace('_', ' ')}: "
            f"{used}/{total} facts discussed ({remaining} unexplored)"
        )
    
    inventory_lines.append("")
    return "\n".join(inventory_lines)


def build_offer_transition_prompt(ex_id: Optional[str], context_section: str,
                                facts_all: List[str], facts_used: List[str],
                                exhibit_names: List[str] = None, knowledge_graph = None,
                                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """
    Build prompt for transitioning to another exhibit.

    Uses exhibit completion tracking to choose the best target exhibit.
    - target_exhibit: The exhibit we want to guide visitor to (from env selection logic)
    - coverage_dict: Museum-wide completion stats (from state tracking)
    """

    current_name = ex_id.replace('_', ' ') if ex_id else 'current exhibit'

    if not target_exhibit:
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {current_name}
---

{context_section}

SummarizeAndSuggest policy:
1. Keep exactly two sentences.
2. Sentence 1: briefly summarize what has been discussed so far.
3. Sentence 2: provide a neutral suggestion to choose any other exhibit.
4. This is only a suggestion; do not imply the visitor already moved.
5. Do not include any [FACT_ID].

Response:"""

    target_name = target_exhibit.replace('_', ' ')

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {current_name}
---

{context_section}

SummarizeAndSuggest policy:
1. Keep exactly two sentences.
2. Sentence 1: briefly summarize what has been discussed so far.
3. Sentence 2: suggest moving to "{target_name}" and name it explicitly.
4. This is only a suggestion; do not imply the visitor already moved.
5. Do not introduce new facts about the target exhibit.
6. Do not include any [FACT_ID].

Response:"""

# ===== CONCLUDE OPTION FUNCTIONS =====

def build_wrap_up_prompt(ex_id: Optional[str], context_section: str,
                        facts_all: List[str], facts_used: List[str],
                        current_completion: float = 0.0) -> str:
    """Build prompt for wrapping up the visit"""
    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

WrapUp policy:
1. Keep exactly two sentences.
2. If the user's latest input contains a clear question:
   - Sentence 1: briefly answer that question directly.
   - Sentence 2: politely close the visit (thanks + warm ending).
3. If there is no clear question:
   - Sentence 1: briefly summarize 1-2 key points already discussed.
   - Sentence 2: politely close the visit (thanks + warm ending).
4. Do not introduce new facts.
5. Do not include any [FACT_ID].
6. Do not ask a new question.
7. Do not mention system state (coverage, policy, strategy, completion).

Response:"""


def build_summarize_key_points_prompt(ex_id: Optional[str], context_section: str,
                                    facts_all: List[str], facts_used: List[str],
                                    current_completion: float = 0.0) -> str:
    """Build prompt for summarizing key points"""
    if facts_used:
        key_points = facts_used[-3:] if len(facts_used) >= 3 else facts_used
        summary_points = "\n".join([f"- {fact}" for fact in key_points])

        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

SummarizeKeyPoints policy:
1. Keep exactly two sentences.
2. Recap 2-3 main points from the key facts below.
3. Do not introduce new facts.
4. Do not include any [FACT_ID].
5. Do not quote the visitor verbatim.

Key facts to summarize:
{summary_points}

Response:"""
    else:
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

SummarizeKeyPoints policy:
1. Keep exactly two sentences.
2. Provide a warm conclusion to the visit.
3. Do not include any [FACT_ID].

Response:"""


