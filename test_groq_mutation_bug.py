#!/usr/bin/env python3
"""
Reproduction script for Bug #11: Groq structured output mutates formatted_messages in-place.

This simulates the exact code flow in llm_invoke.py lines 1865-2903 
WITHOUT making any actual API calls. It shows how Groq's schema injection
permanently corrupts the prompt for all subsequent model candidates.

Run: python test_groq_mutation_bug.py
"""

import json
from copy import deepcopy

# ─── Colors for terminal output ───
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def simulate_llm_invoke_bug():
    """Simulate the model retry loop from llm_invoke.py with the Groq mutation bug."""

    print(f"\n{BOLD}{'='*70}")
    print(f"  Bug #11: Groq Message Mutation Reproduction")
    print(f"{'='*70}{RESET}\n")

    # ─── Setup: this is what formatted_messages looks like entering the loop ───
    formatted_messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a function that adds two numbers."},
    ]

    # Simulate candidate models (typical strength < 0.5 ordering)
    candidate_models = [
        {"model": "groq/llama-3.1-70b", "provider": "groq"},
        {"model": "gpt-4o-mini", "provider": "openai"},
        {"model": "claude-3-haiku", "provider": "anthropic"},
    ]

    # Groq's schema instruction (simplified from the real code)
    schema_instruction = (
        'You must respond with valid JSON matching this schema:\n'
        '```json\n{"type": "object", "properties": {"code": {"type": "string"}}}\n```\n'
        'Respond ONLY with the JSON object, no other text.'
    )

    is_groq_candidate = lambda m: m["provider"] == "groq"

    # ─── Show original state ───
    print(f"{CYAN}📋 Original formatted_messages:{RESET}")
    print(f"   System: {DIM}\"{formatted_messages[0]['content']}\"{RESET}")
    print(f"   User:   {DIM}\"{formatted_messages[1]['content']}\"{RESET}")
    print(f"   {DIM}id(formatted_messages) = {id(formatted_messages)}{RESET}")
    print()

    # ─── Simulate the model retry loop (lines 1865-2903 of llm_invoke.py) ───
    for i, model_info in enumerate(candidate_models):
        model_name = model_info["model"]
        provider = model_info["provider"]

        print(f"{BOLD}{'─'*70}")
        print(f"  Attempt {i+1}: {model_name} ({provider})")
        print(f"{'─'*70}{RESET}")

        # This is what llm_invoke does at line 1888
        litellm_kwargs = {
            "model": model_name,
            "messages": formatted_messages,  # ← DIRECT REFERENCE, not a copy!
            "temperature": 0.1,
        }

        # ─── Groq structured output handling (lines 2086-2108) ───
        if is_groq_candidate(model_info):
            print(f"\n   {YELLOW}⚠  Groq detected → injecting schema into system message{RESET}")

            # THIS IS THE BUG (line 2100):
            messages_list = litellm_kwargs.get("messages", [])  # ← alias, not copy!
            
            print(f"   {DIM}messages_list is formatted_messages: {messages_list is formatted_messages}{RESET}")
            
            if messages_list and messages_list[0].get("role") == "system":
                # Line 2102: mutates the dict IN-PLACE
                messages_list[0]["content"] = schema_instruction + "\n\n" + messages_list[0]["content"]
            else:
                # Line 2104: mutates the list IN-PLACE
                messages_list.insert(0, {"role": "system", "content": schema_instruction})
            litellm_kwargs["messages"] = messages_list

            print(f"   {RED}💥 System message MUTATED:{RESET}")
            print(f"      {DIM}\"{messages_list[0]['content'][:80]}...\"{RESET}")

            # Simulate Groq failing (e.g., rate limit)
            print(f"\n   {RED}✗ Groq call failed: RateLimitError{RESET}")
            print(f"   {DIM}→ breaking inner loop, trying next model...{RESET}")
            continue  # Next model candidate

        else:
            # ─── Non-Groq models see the corrupted messages ───
            current_system = litellm_kwargs["messages"][0]["content"]
            
            if "Respond ONLY with the JSON object" in current_system:
                print(f"\n   {RED}🚨 CORRUPTED! System message contains Groq schema preamble:{RESET}")
                # Show the corruption clearly
                lines = current_system.split('\n')
                for line in lines[:4]:
                    print(f"      {RED}{line}{RESET}")
                print(f"      {RED}...{RESET}")
                print(f"      {RED}{lines[-1]}{RESET}")
            else:
                print(f"\n   {GREEN}✓ System message is clean{RESET}")

            print(f"\n   {DIM}(Would call litellm.completion() here with corrupted messages){RESET}")
            # Don't actually call — just demonstrating the state
            continue

    # ─── Final state ───
    print(f"\n{BOLD}{'='*70}")
    print(f"  Final State")
    print(f"{'='*70}{RESET}")
    print(f"\n{RED}📋 formatted_messages is NOW permanently corrupted:{RESET}")
    print(f"   System message length: {len(formatted_messages[0]['content'])} chars")
    print(f"   {DIM}(was originally 42 chars){RESET}")
    print(f"\n   Full system message:")
    for line in formatted_messages[0]["content"].split('\n'):
        print(f"   {RED}│ {line}{RESET}")


def simulate_fixed_version():
    """Show how the fix prevents the mutation."""

    print(f"\n\n{BOLD}{'='*70}")
    print(f"  FIXED Version: Using a copy")
    print(f"{'='*70}{RESET}\n")

    formatted_messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a function that adds two numbers."},
    ]

    schema_instruction = (
        'You must respond with valid JSON matching this schema:\n'
        '```json\n{"type": "object", "properties": {"code": {"type": "string"}}}\n```\n'
        'Respond ONLY with the JSON object, no other text.'
    )

    candidate_models = [
        {"model": "groq/llama-3.1-70b", "provider": "groq"},
        {"model": "gpt-4o-mini", "provider": "openai"},
    ]

    for i, model_info in enumerate(candidate_models):
        model_name = model_info["model"]
        provider = model_info["provider"]

        print(f"{BOLD}{'─'*70}")
        print(f"  Attempt {i+1}: {model_name}")
        print(f"{'─'*70}{RESET}")

        litellm_kwargs = {
            "model": model_name,
            "messages": formatted_messages,
            "temperature": 0.1,
        }

        if provider == "groq":
            # ─── THE FIX: copy the list and its dicts ───
            messages_list = [dict(m) for m in litellm_kwargs.get("messages", [])]
            
            print(f"   {GREEN}✓ messages_list is formatted_messages: {messages_list is formatted_messages}{RESET}")

            if messages_list and messages_list[0].get("role") == "system":
                messages_list[0]["content"] = schema_instruction + "\n\n" + messages_list[0]["content"]
            litellm_kwargs["messages"] = messages_list

            print(f"   {YELLOW}⚠  Groq schema injected into COPY only{RESET}")
            print(f"   {RED}✗ Groq call failed: RateLimitError{RESET}")
            continue

        else:
            current_system = litellm_kwargs["messages"][0]["content"]
            if "Respond ONLY with the JSON object" in current_system:
                print(f"\n   {RED}🚨 CORRUPTED!{RESET}")
            else:
                print(f"\n   {GREEN}✓ System message is CLEAN — original preserved!{RESET}")
                print(f"      \"{current_system}\"")


if __name__ == "__main__":
    simulate_llm_invoke_bug()
    simulate_fixed_version()
    print()
