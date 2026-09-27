#!/usr/bin/env python
"""Stream a real multi-turn conversation and print authoritative server timings.

python examples/chat_http.py --base-url http://127.0.0.1:8080
python examples/chat_http.py --prompt 'Explain what a KV cache is.' --max-tokens 128
This is a CPU HTTP client; the API server must already be ready to serve requests.
"""
import argparse
import json
import time

import anthropic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    parser.add_argument('--model', default='naive-n0.5')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--prompt', action='append', help='Repeat for successive conversation turns')
    args = parser.parse_args()
    prompts = args.prompt or ['Remember my code word: ZEBRA-7. Briefly confirm.', 'What is my code word? Reply with only the code word.']
    messages = []
    # Use the SDK's default retry policy.
    with anthropic.Anthropic(base_url=args.base_url, api_key='x', timeout=600) as client:
        for prompt in prompts:
            print(f'\nyou> {prompt}\nnaive-n> ', end='', flush=True)
            started = time.perf_counter()
            first = None
            timings = None
            with client.messages.stream(model=args.model, max_tokens=args.max_tokens,
                                         messages=[*messages, {'role':'user','content':prompt}]) as stream:
                for event in stream:
                    if event.type == 'content_block_delta' and event.delta.type == 'text_delta':
                        if event.delta.text and first is None:
                            first = time.perf_counter() - started
                        print(event.delta.text, end='', flush=True)
                    elif event.type == 'message_delta':
                        timings = (event.model_extra or {}).get('naivert', {}).get('timings')
                reply = stream.get_final_message()
            elapsed = time.perf_counter() - started
            text = ''.join(block.text for block in reply.content)
            messages.extend([{'role':'user','content':prompt}, {'role':'assistant','content':text}])
            print('\n' + json.dumps({'ttft_seconds':first, 'total_seconds':elapsed,
                                    'usage':reply.usage.model_dump(exclude_none=True),
                                    'server_timings':timings}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
