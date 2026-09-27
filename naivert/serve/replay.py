"""Bounded, exact-history restoration of assistant whitespace omitted by a client.

No whitespace in user text, tool arguments, or nonempty blocks is normalized.
Only the known projection of a response previously sent by this server can match.
This HTTP-side metadata never grants a KV hit: the engine still compares token IDs.
"""
from collections import OrderedDict
import hashlib
import json
import threading

from .messages_api import _assistant_message


def encoded(value):
    # Preserve object order: tool parameter order affects the model's prompt.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def advance(digest, value):
    data = encoded(value)
    digest.update(len(data).to_bytes(8, 'big'))
    digest.update(data)


class AssistantReplay:
    def __init__(self, *, max_bytes=16 << 20, max_entries=256):
        self.entries = OrderedDict()
        self.bytes = 0
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.lock = threading.Lock()

    def prefix(self, task):
        digest = hashlib.sha256()
        advance(digest, [task.session_id, task.tools, task.enable_thinking])
        return digest

    def restore(self, task):
        if task.messages is None:
            return
        digest = self.prefix(task)
        restored = []
        for message in task.messages:
            if message.get('role') == 'assistant':
                key = digest.copy()
                advance(key, message)
                with self.lock:
                    record = self.entries.get(key.digest())
                    if record is not None:
                        self.entries.move_to_end(key.digest())
                if record is not None and record[0] is not None:
                    message = json.loads(record[0])
            restored.append(message)
            advance(digest, message)
        task.messages = restored

    def remember(self, task, blocks):
        if task.messages is None:
            return
        kept = [b for b in blocks if b['type'] not in ('thinking', 'text')
                or b.get('thinking', b.get('text', '')).strip()]
        original, _ = _assistant_message(blocks)
        projected, _ = _assistant_message(kept)
        # Remember identity projections too: a retry with no blank blocks can
        # otherwise be confused with an earlier answer whose blanks were dropped.
        digest = self.prefix(task)
        for message in task.messages:
            advance(digest, message)
        advance(digest, projected)
        key, data = digest.digest(), encoded(original)
        size = len(key) + len(data)
        if size > self.max_bytes:
            return
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
                if old[0] != data:
                    # Two outputs differ only in omitted whitespace: do not guess which
                    # one the client is replaying. Retain a bounded ambiguity marker.
                    data, size = None, len(key)
            self.entries[key] = (data, size)
            self.bytes += size
            while self.entries and (self.bytes > self.max_bytes or len(self.entries) > self.max_entries):
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
