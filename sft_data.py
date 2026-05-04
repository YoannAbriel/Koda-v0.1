"""
SFT data preparation: convert Dolly + OASST to unified chat format.

We use a Llama-style chat template with simple text markers
(no special tokens — we work with the existing GPT-2 vocabulary):

  <|user|>
  {user message}
  <|assistant|>
  {assistant message}
  <|end|>

This template is simple, works with GPT-2 tokenizer, and clearly
marks turns so the model can learn to switch roles.
"""

from datasets import load_dataset
import random

USER_TAG = '<|user|>'
ASSISTANT_TAG = '<|assistant|>'
END_TAG = '<|end|>'


def format_conversation(messages):
    """Convert a list of {role, content} dicts into a single string.

    Example:
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!'},
        ]
        →
        '<|user|>\nHello\n<|assistant|>\nHi!\n<|end|>'
    """
    parts = []
    for msg in messages:
        if msg['role'] == 'user':
            parts.append(f'{USER_TAG}\n{msg["content"]}')
        elif msg['role'] == 'assistant':
            parts.append(f'{ASSISTANT_TAG}\n{msg["content"]}')
    parts.append(END_TAG)
    return '\n'.join(parts)


def dolly_to_messages(example):
    """Convert a Dolly example to chat messages."""
    instruction = example['instruction'].strip()
    context = example.get('context', '').strip()
    response = example['response'].strip()

    # If there's context, prepend it to the instruction
    if context:
        user_content = f'{instruction}\n\nContext:\n{context}'
    else:
        user_content = instruction

    return [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': response},
    ]


def oasst_to_conversations(ds):
    """Convert OASST tree structure to flat conversation lists.

    For each leaf assistant message, walk up the tree to root
    to reconstruct the full conversation.
    """
    # Build a map of message_id → message
    msg_by_id = {ex['message_id']: ex for ex in ds}

    conversations = []
    for ex in ds:
        # Only build conversations ending with an assistant message
        if ex['role'] != 'assistant':
            continue
        if ex['lang'] != 'en':
            continue

        # Walk up the tree to root
        chain = []
        current = ex
        while current is not None:
            chain.append(current)
            parent_id = current.get('parent_id')
            if parent_id and parent_id in msg_by_id:
                current = msg_by_id[parent_id]
            else:
                break

        chain.reverse()  # root → leaf

        # Convert to message dicts
        messages = []
        for m in chain:
            role = 'user' if m['role'] == 'prompter' else 'assistant'
            messages.append({'role': role, 'content': m['text']})

        # Skip if conversation doesn't start with user
        if messages and messages[0]['role'] != 'user':
            continue
        # Skip if too short
        if len(messages) < 2:
            continue

        conversations.append(messages)

    return conversations


def main():
    print('=== Loading Dolly ===')
    dolly = load_dataset('databricks/databricks-dolly-15k', split='train')
    dolly_msgs = [dolly_to_messages(ex) for ex in dolly]
    print(f'Dolly conversations: {len(dolly_msgs):,}')

    print('\n=== Loading OASST ===')
    oasst = load_dataset('OpenAssistant/oasst1', split='train')
    oasst_msgs = oasst_to_conversations(oasst)
    print(f'OASST conversations: {len(oasst_msgs):,}')

    # Combine
    all_convs = dolly_msgs + oasst_msgs
    print(f'\nTotal: {len(all_convs):,}')

    # Show 3 formatted examples from each
    print('\n=== DOLLY FORMATTED EXAMPLES ===')
    for i in range(2):
        text = format_conversation(dolly_msgs[i])
        print(f'\n--- Dolly {i+1} ---')
        print(text[:600])

    print('\n=== OASST FORMATTED EXAMPLES ===')
    random.seed(42)
    for i, conv in enumerate(random.sample(oasst_msgs, 2)):
        text = format_conversation(conv)
        print(f'\n--- OASST {i+1} (turns: {len(conv)}) ---')
        print(text[:600])

    # Multi-turn count
    multi_turn_oasst = sum(1 for c in oasst_msgs if len(c) > 2)
    print(f'\n=== STATS ===')
    print(f'OASST multi-turn (>2 messages): {multi_turn_oasst:,} ({multi_turn_oasst/len(oasst_msgs)*100:.0f}%)')
    print(f'Dolly multi-turn: 0 (all single Q/A)')


if __name__ == '__main__':
    main()
