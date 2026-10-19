"""Check real generation for baseline, eight task directions and a combination.

Run against a ready backend, not as a substitute for evaluating steering quality.
"""
import argparse
import json
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18000')
    arguments = parser.parse_args()
    base_url = arguments.url.rstrip('/')
    health = requests.get(f'{base_url}/health', timeout=10)
    health.raise_for_status()
    cases = [('baseline', {})]
    for task_name in ['sentiment', 'toxicity', 'debate', 'polite']:
        for direction in ['pos-neg', 'neg-pos']:
            name = f'{task_name}_{direction}'
            cases.append((name, {name: 1.0}))
    cases.append(('combined', {'sentiment_pos-neg': 1.0, 'polite_pos-neg': 1.0}))
    for case_name, strengths in cases:
        started = time.monotonic()
        response = requests.post(f'{base_url}/process', json={
            'input_dict': strengths, 'input_str': 'The movie was',
        }, timeout=180)
        response.raise_for_status()
        result = response.json()['result']
        if not isinstance(result, str) or not result.startswith('The movie was') or len(result) <= 13:
            raise AssertionError(f'{case_name}: invalid generation {result!r}')
        print(json.dumps({'case': case_name, 'characters': len(result),
                          'seconds': round(time.monotonic()-started, 2)}), flush=True)


if __name__ == '__main__':
    main()
