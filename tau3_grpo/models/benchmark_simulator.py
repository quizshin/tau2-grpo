"""Measure a local simulator with recorded chat prompts; retain every response."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


async def measure(args):
    prompts = json.loads(args.prompts.read_text())
    if not prompts:
        raise ValueError('At least one prompt is required')
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        async def request(prompt, concurrency, index, semaphore):
            async with semaphore:
                started = time.perf_counter()
                first = None
                content, reasoning, usage, finish = '', '', {}, None
                async with client.stream('POST', args.base_url.rstrip('/') + '/chat/completions', json={
                    'model': args.model, 'messages': prompt['messages'],
                    'temperature': 0, 'max_tokens': args.max_tokens,
                    'stream': True, 'stream_options': {'include_usage': True},
                    'chat_template_kwargs': {'enable_thinking': False},
                }) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: ') or line == 'data: [DONE]':
                            continue
                        chunk = json.loads(line[6:])
                        if chunk.get('usage'):
                            usage = chunk['usage']
                        for choice in chunk.get('choices', []):
                            delta = choice.get('delta', {})
                            token = delta.get('content') or ''
                            thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                            if (token or thought) and first is None:
                                first = time.perf_counter() - started
                            content += token
                            reasoning += thought
                            finish = choice.get('finish_reason') or finish
                row = {'concurrency': concurrency, 'index': index, 'prompt_id': prompt.get('id'),
                       'seconds': time.perf_counter() - started, 'ttft_seconds': first,
                       'usage': usage, 'finish_reason': finish, 'content': content,
                       'reasoning': reasoning}
                rows.append(row)
                with args.output.with_suffix('.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                if not content.strip() or reasoning.strip() or '<think>' in content:
                    raise ValueError('Empty response or unexpected thinking; see recorded response')
                if 'completion_tokens' not in usage:
                    raise ValueError('Server did not return token usage')
                return row

        # First request warms the model; it is recorded separately from measured rounds.
        await request(prompts[0], 0, 0, asyncio.Semaphore(1))
        summaries = []
        for concurrency in args.concurrency:
            if concurrency < 1:
                raise ValueError('Concurrency must be positive')
            semaphore = asyncio.Semaphore(concurrency)
            start = time.perf_counter()
            batch = await asyncio.gather(*(request(p, concurrency, i, semaphore)
                                           for i, p in enumerate(prompts)))
            seconds = time.perf_counter() - start
            latencies = sorted(row['seconds'] for row in batch)
            summary = {'concurrency': concurrency, 'requests': len(batch), 'wall_seconds': seconds,
                       'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in batch) / seconds,
                       'requests_per_second': len(batch) / seconds,
                       'latency_p50_seconds': latencies[math.ceil(len(batch) * .5) - 1],
                       'latency_p95_seconds': latencies[math.ceil(len(batch) * .95) - 1],
                       'ttft_mean_seconds': sum(r['ttft_seconds'] for r in batch) / len(batch),
                       'truncated_responses': sum(r['finish_reason'] == 'length' for r in batch)}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            args.output.write_text(json.dumps({'model': args.model, 'base_url': args.base_url,
                                               'max_tokens': args.max_tokens, 'rounds': summaries,
                                               'note': 'Repeated prompts and prefix cache: warm-cache throughput.'},
                                              indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8100/v1')
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B-AWQ-INT4')
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        parser.error('Choose a new output path; existing measurements are preserved')
    asyncio.run(measure(args))


if __name__ == '__main__':
    main()
