#!/usr/bin/env python3
"""Run the development-only frozen BEATs recording-label comparison."""

import argparse
import json
from pathlib import Path

from drone_detector.beats_encoder import BEATsEncoder
from drone_detector.frozen_probe import ProbeConfig, run_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reference-bundle', type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--encoder-family', default='beats', choices=['beats', 'clap'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--window-seconds', type=float, default=3)
    parser.add_argument('--hop-seconds', type=float, default=.5)
    parser.add_argument('--min-audio-seconds', type=float, default=.25)
    parser.add_argument('--max-windows', type=int, default=60)
    parser.add_argument('--channel-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regularization-c', type=float, default=1)
    args = parser.parse_args()
    config = ProbeConfig(window_seconds=args.window_seconds, hop_seconds=args.hop_seconds,
                         min_audio_seconds=args.min_audio_seconds, max_windows=args.max_windows,
                         channel_index=args.channel_index, seed=args.seed,
                         regularization_c=args.regularization_c)
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    if args.encoder_family == 'beats':
        encoder = BEATsEncoder(args.encoder, device=args.device)
    else:
        from drone_detector.clap_encoder import CLAPEncoder
        encoder = CLAPEncoder(args.encoder, device=args.device)
    encoder.identity['torch_intraop_threads'] = args.threads
    report = run_probe(args.manifest, args.output, args.cache, encoder, config,
                       reference_bundle=args.reference_bundle)
    print(json.dumps({'model_id': report['model_id'], 'output': str(args.output),
                      'training_recordings': report['training_recordings'],
                      'development_recordings': report['development_recordings'],
                      'cache_hits': report['cache_hits'],
                      'scope': report['scope']}, indent=2))


if __name__ == '__main__':
    main()
