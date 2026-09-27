"""Compare private Claude request pairs; emit hashes and structure, never bodies or headers."""
import argparse
import hashlib
import json
from pathlib import Path


def normalize(value):
    if isinstance(value, list):
        return [normalize(x) for x in value]
    if isinstance(value, dict):
        result = {k: normalize(v) for k, v in value.items() if k != 'cache_control'}
        if isinstance(result.get('content'), str):
            result['content'] = [dict(type='text', text=result['content'])]
        return result
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--pair', action='append', required=True, help='earlier_request_id:later_request_id')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pairs = [x.split(':', 1) for x in a.pair]
    wanted = {rid for pair in pairs for rid in pair}
    requests, responses = {}, {}
    digest = hashlib.sha256()
    with a.capture.open('rb') as source:
        for line in source:
            digest.update(line)
            row = json.loads(line)
            if row.get('rid') not in wanted:
                continue
            if row.get('response'):
                responses[row['rid']] = row.get('usage')
            elif row.get('body'):
                requests[row['rid']] = json.loads(row['body'])
    result = dict(capture_sha256=digest.hexdigest(), pairs=[],
                  scope='Structural comparison after stripping cache hints and normalizing text blocks; sampled token IDs are unavailable, so this does not prove a historical token mismatch cause.')
    for old, new in pairs:
        left, right = requests[old], requests[new]
        n = len(left['messages'])
        extra = right['messages'][n:]
        result['pairs'].append(dict(previous=old, next=new,
            previous_messages=n, next_messages=len(right['messages']),
            prior_messages_equal=normalize(left['messages']) == normalize(right['messages'][:n]),
            system_equal=normalize(left.get('system')) == normalize(right.get('system')),
            tools_equal=normalize(left.get('tools')) == normalize(right.get('tools')),
            next_usage=responses.get(new),
            added_assistant_blocks=[[b.get('type') for b in m['content']]
                                    for m in extra if m['role'] == 'assistant' and isinstance(m['content'], list)]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
