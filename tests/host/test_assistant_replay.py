from copy import deepcopy

import pytest

from naivert.serve.messages_api import ApiError, _assistant_message
from naivert.serve.replay import AssistantReplay
from naivert.serve.task import ServerTask


def task(messages=None, **kwargs):
    return ServerTask(id=1, messages=messages or [{'role': 'user', 'content': 'hello'}],
                      session_id=kwargs.pop('session_id', 'parent'), enable_thinking=True, **kwargs)


def assistant(blocks):
    return _assistant_message(blocks)[0]


@pytest.mark.parametrize('blocks', [
    [{'type': 'thinking', 'thinking': '\n\t '}, {'type': 'text', 'text': 'answer'}],
    [{'type': 'text', 'text': '\n '}, {'type': 'thinking', 'thinking': 'plan'}, {'type': 'text', 'text': 'answer'}],
    [{'type': 'thinking', 'thinking': 'plan'}, {'type': 'text', 'text': '\n\n\t'},
     {'type': 'tool_use', 'id': 'one', 'name': 'read', 'input': {'path': ' a\n b '}},
     {'type': 'text', 'text': '\n\n '},
     {'type': 'tool_use', 'id': 'two', 'name': 'read', 'input': {'path': 'second'}},
     {'type': 'text', 'text': '\n\t'}],
])
def test_restores_only_observed_omitted_blocks(blocks):
    cache = AssistantReplay()
    first = task()
    cache.remember(first, blocks)
    kept = [b for b in blocks if b['type'] not in ('thinking', 'text') or b.get('thinking', b.get('text', '')).strip()]
    follow = task(first.messages + [assistant(kept), {'role': 'user', 'content': 'next'}])
    original_input = deepcopy(follow.messages)
    cache.restore(follow)
    assert follow.messages[1] == assistant(blocks)
    assert original_input[1] == assistant(kept)  # no mutation of the caller's dictionaries
    assert follow.messages[0] == first.messages[0]
    assert follow.messages[-1] == original_input[-1]


def test_history_scope_changed_content_and_explicit_whitespace_do_not_match():
    cache = AssistantReplay()
    blocks = [{'type': 'thinking', 'thinking': '\n'}, {'type': 'text', 'text': 'answer'}]
    cache.remember(task(), blocks)
    for changed in [
        task([{'role': 'user', 'content': 'different'}, assistant(blocks[1:])]),
        task([{'role': 'user', 'content': 'hello'}, assistant(blocks[1:])], session_id='other'),
        task([{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'edited'}]),
        task([{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'answer', 'reasoning_content': '  '}]),
        task([{'role': 'user', 'content': 'hello'}, assistant(blocks[1:])], tools=[{'name': 'new'}]),
    ]:
        before = deepcopy(changed.messages)
        cache.restore(changed)
        assert changed.messages == before


def test_multiple_turns_restore_sequentially_and_ambiguous_outputs_fail_closed():
    cache = AssistantReplay()
    blocks = [{'type': 'thinking', 'thinking': '\n'}, {'type': 'text', 'text': 'answer'}]
    first = task()
    cache.remember(first, blocks)
    second = task(first.messages + [assistant(blocks), {'role': 'user', 'content': 'again'}])
    cache.remember(second, blocks)
    replay = task(first.messages + [assistant(blocks[1:]), second.messages[-1], assistant(blocks[1:])])
    cache.restore(replay)
    assert replay.messages[1] == replay.messages[3] == assistant(blocks)
    cache.remember(first, [{'type': 'thinking', 'thinking': '  '}, blocks[1]])
    follow = task(first.messages + [assistant(blocks[1:])])
    cache.restore(follow)
    assert follow.messages[1] == assistant(blocks[1:])


def test_bounded_storage_and_tool_text_order():
    cache = AssistantReplay(max_entries=1, max_bytes=1024)
    blocks = [{'type': 'thinking', 'thinking': '\n'}, {'type': 'text', 'text': 'answer'}]
    cache.remember(task(), blocks)
    cache.remember(task(session_id='other'), blocks)
    assert len(cache.entries) == 1 and cache.bytes <= 1024
    cache.remember(task(), [{'type': 'thinking', 'thinking': ' ' * 2000}, blocks[1]])
    assert len(cache.entries) == 1 and cache.bytes <= 1024
    call = {'type': 'tool_use', 'id': 'one', 'name': 'read', 'input': {'path': 'x'}}
    content = assistant([{'type': 'text', 'text': 'before'}, call, {'type': 'text', 'text': 'after'}])['content']
    assert content.startswith('before<tool_call>') and content.endswith('</tool_call>after')


@pytest.mark.parametrize('blank_first', [False, True])
def test_identity_projection_also_makes_a_retry_ambiguous(blank_first):
    cache = AssistantReplay()
    plain = [{'type': 'text', 'text': 'answer'}]
    blank = [{'type': 'thinking', 'thinking': '\n'}, *plain]
    for blocks in ([blank, plain] if blank_first else [plain, blank]):
        cache.remember(task(), blocks)
    follow = task(task().messages + [assistant(plain)])
    cache.restore(follow)
    assert follow.messages[-1] == assistant(plain)


def test_sdk_parsed_output_null_is_inert_but_nonnull_is_not_silently_dropped():
    assert assistant([{'type': 'text', 'text': 'x', 'parsed_output': None}])['content'] == 'x'
    with pytest.raises(ApiError, match='parsed_output'):
        assistant([{'type': 'text', 'text': 'x', 'parsed_output': {'x': 1}}])
