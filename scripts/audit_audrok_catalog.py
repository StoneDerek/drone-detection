#!/usr/bin/env python3
"""Inspect an AuDroK MySQL text dump without executing any SQL.

This deliberately supports only the dump's CREATE-column and INSERT-VALUES
syntax. Unknown literal forms stop parsing. Metadata rows are not recordings,
and their durations are not de-duplicated capture exposure.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
ESCAPES = {'0': '\0', 'b': '\b', 'n': '\n', 'r': '\r', 't': '\t', 'Z': '\x1a'}


def _values(text):
    position, rows = 0, []
    while position < len(text):
        if text[position] != '(':
            raise ValueError('Expected a VALUES row')
        position += 1
        row = []
        while True:
            while text[position].isspace():
                position += 1
            if text[position] == "'":
                position += 1
                parts = []
                while True:
                    char = text[position]
                    position += 1
                    if char == '\\':
                        escaped = text[position]
                        position += 1
                        parts.append(ESCAPES.get(escaped, escaped))
                    elif char == "'":
                        if position < len(text) and text[position] == "'":
                            parts.append("'")
                            position += 1
                        else:
                            break
                    else:
                        parts.append(char)
                value = ''.join(parts)
            elif text.startswith('NULL', position):
                value, position = None, position + 4
            else:
                match = NUMBER.match(text, position)
                if match is None:
                    raise ValueError('Unsupported SQL literal; do not execute it')
                token = match.group()
                value = float(token) if any(c in token for c in '.eE') else int(token)
                position = match.end()
            row.append(value)
            while text[position].isspace():
                position += 1
            char, position = text[position], position + 1
            if char == ')':
                break
            if char != ',':
                raise ValueError('Expected a VALUES separator')
        rows.append(row)
        if position == len(text):
            break
        if text[position] != ',':
            raise ValueError('Unexpected text after VALUES row')
        position += 1
    return rows


def values(text):
    try:
        return _values(text)
    except IndexError as error:
        raise ValueError('Truncated SQL VALUES input') from error


def parse_catalog(text):
    tables = {}
    for match in re.finditer(r'^CREATE TABLE `([^`]+)` \(\n(.*?)\n\) ENGINE=', text, re.M | re.S):
        columns = re.findall(r'^  `([^`]+)` ', match[2], re.M)
        if not columns:
            raise ValueError('Table with no recognized columns')
        if match[1] in tables:
            raise ValueError('Repeated table schema')
        tables[match[1]] = {'columns': columns, 'rows': []}
    insert_lines = [line for line in text.splitlines() if line.startswith('INSERT ')]
    for line in insert_lines:
        match = re.fullmatch(r'INSERT INTO `([^`]+)` VALUES (.*);', line)
        if match is None or match[1] not in tables:
            raise ValueError('Unsupported INSERT form/table')
        table = tables[match[1]]
        for row in values(match[2]):
            if len(row) != len(table['columns']):
                raise ValueError('SQL row/column count differs')
            table['rows'].append(dict(zip(table['columns'], row, strict=True)))
    if not insert_lines or not tables:
        raise ValueError('No catalog tables/values recognized')
    return tables


def summarize(tables):
    results = []
    for name, table in tables.items():
        rows = table['rows']
        if 'ReplayTime_Sek' not in table['columns']:
            continue
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row['Origin']].append(row)
        for origin, items in groups.items():
            durations = [row['ReplayTime_Sek'] for row in items if isinstance(row['ReplayTime_Sek'], (int, float))]
            results.append({'table': name, 'origin': origin, 'metadata_rows': len(items),
                            'rows_with_duration': len(durations),
                            'summed_file_seconds': sum(durations),
                            'drone_type_ids': sorted({row['DroneType'] for row in items if row['DroneType'] is not None})})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sql', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tables = parse_catalog(args.sql.read_text(encoding='utf-8'))
    report = {'scope': 'Metadata only, SQL never executed; file-duration sums are not unique timeline hours',
              'catalog_sha256': hashlib.sha256(args.sql.read_bytes()).hexdigest(),
              'tables': {name: len(table['rows']) for name, table in tables.items()},
              'by_table_origin': summarize(tables)}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'tables.json').write_text(json.dumps(tables, ensure_ascii=False) + '\n')
    print(json.dumps({'tables': report['tables'], 'grouped_rows': len(report['by_table_origin'])}, indent=2))


if __name__ == '__main__':
    main()
