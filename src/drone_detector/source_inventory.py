"""Conservative recording/view accounting for the renamed AuDroK outdoor files."""

import collections
import hashlib
import json
import re
from pathlib import PurePosixPath

NAME = re.compile(r'(?P<measurement>\d+)-(?P<drone>\d+)-(?P<condition>(?:[SF]-\d+m)|var)-(?P<view>Arr[123]|Mic1)\.wav')


def outdoor_name(path):
    name = PurePosixPath(path.replace('\\', '/')).name
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError(f'Unrecognized outdoor recording name: {name}')
    fields = match.groupdict()
    return {'measurement_key': name.rsplit('-', 1)[0],
            'publisher_measurement_number': int(fields['measurement']),
            'publisher_drone_code': int(fields['drone']),
            'publisher_condition_code': fields['condition'], 'view': fields['view'],
            'input_kind': 'point_microphone' if fields['view'] == 'Mic1' else 'array_mean_24_microphones'}


def summarize_views(rows):
    groups = collections.defaultdict(list)
    identities = set()
    for row in rows:
        identity = (row['measurement_key'], row['view'])
        if identity in identities:
            raise ValueError('Repeated measurement/view across archives; audit mirrors before counting')
        identities.add(identity)
        groups[row['measurement_key']].append(row)
    measurements = []
    for key, members in sorted(groups.items()):
        if {row['view'] for row in members} != {'Arr1', 'Arr2', 'Arr3', 'Mic1'}:
            raise ValueError(f'Missing recorder views in {key}')
        if len({row['complete_frames'] for row in members}) != 1 or len({row['sample_rate'] for row in members}) != 1:
            raise ValueError(f'Recorder durations/rates differ in {key}')
        point = next(row for row in members if row['view'] == 'Mic1')
        measurements.append({'measurement_key': key,
            'publisher_measurement_number': point['publisher_measurement_number'],
            'publisher_drone_code': point['publisher_drone_code'],
            'publisher_condition_code': point['publisher_condition_code'],
            'point_microphone_file_seconds': point['duration_seconds'],
            'recorder_views': len(members)})
    return {'wav_files': len(rows), 'measurement_groups': len(measurements),
            'point_microphone_files': sum(row['input_kind'] == 'point_microphone' for row in rows),
            'array_mean_files': sum(row['input_kind'] == 'array_mean_24_microphones' for row in rows),
            'all_view_file_seconds': sum(row['duration_seconds'] for row in rows),
            'point_microphone_file_seconds': sum(row['point_microphone_file_seconds'] for row in measurements),
            'publisher_drone_codes': sorted({row['publisher_drone_code'] for row in measurements}),
            'condition_file_seconds': dict(sorted((code, sum(row['point_microphone_file_seconds']
                for row in measurements if row['publisher_condition_code'] == code))
                for code in {row['publisher_condition_code'] for row in measurements})),
            'measurements': measurements,
            'scope': 'File durations only, one campaign. Publisher codes are retained without inventing flight-state or audible-event labels. Array means are not individual microphones.'}


def validate_completion_receipt(item, receipt):
    identity = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (receipt['item_identity'] != identity or receipt['filename'] != item['filename']
            or receipt['bytes'] != item['bytes'] or receipt['url'] != item['url']
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256'])
            or (item['sha256'] is not None and receipt['sha256'] != item['sha256'])):
        raise ValueError('Completion receipt differs from the pinned intake plan')
