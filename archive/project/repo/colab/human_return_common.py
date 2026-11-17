"""Pure, stdlib-only guards shared by human output recovery tools."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

CLOUD_ROOT = '/content/DynaVelo_Research'
GENERATION = 'dynavelo-hspc-pilot-20261112'
RUN_RE = re.compile(r'8489-MV-1_cuda_seed[012]_\d{8}T\d{12}Z')
SHA_RE = re.compile(r'[0-9a-f]{64}')
RESERVED = re.compile(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', re.I)
MAX_CORE_BYTES = 256 * 1024**2

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def capture_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]

def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Unsafe namespace component')
    safe_relative(value)
    return value

def safe_relative(value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(char) < 32 or char in '<>"|?*' for char in value)):
        raise ValueError('Unsafe relative path')
    parts = value.split('/')
    if (any(part in {'', '.', '..'} or part.endswith(('.', ' ')) or RESERVED.fullmatch(part) for part in parts)
            or PurePosixPath(value).is_absolute()):
        raise ValueError('Unsafe relative path component')
    return value

def safe_target(root, relative):
    safe_relative(relative)
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError('Project root is redirected')
    target = root.joinpath(*relative.split('/'))
    if target.resolve() != target or not target.is_relative_to(root):
        raise ValueError('Symlink/junction or escaping destination')
    return target

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def byte_record(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')

def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def namespace_path(source, generation, capture):
    safe_relative(source)
    safe_component(generation); safe_component(capture)
    for prefix in ('checkpoints/', 'data/processed/', 'results/reports/'):
        if source.startswith(prefix):
            return prefix + 'cloud_returns/' + generation + '/' + capture + '/' + source[len(prefix):]
    raise ValueError('Source outside output roots')

def validate_run(run_id):
    if not isinstance(run_id, str) or not RUN_RE.fullmatch(run_id):
        raise ValueError('Unexpected human run identifier')
    return run_id

def validate_restore(raw, run_id):
    validate_run(run_id)
    if len(raw) > 2 * 1024**2:
        raise ValueError('Recovery manifest too large')
    value = json.loads(raw.decode('utf-8-sig'))
    if (value.get('schema') != 'dynavelo_colab_checkpoint_restore_v1'
            or value.get('cloud_root') != CLOUD_ROOT or value.get('run_id') != run_id
            or type(value.get('completed_epoch')) is not int or value['completed_epoch'] < 1):
        raise ValueError('Recovery manifest identity/schema mismatch')
    expected = {'checkpoint': {'last_checkpoint.pth'}, 'prior_config': {'resume_config.json'},
                'best_snapshot': {'best_snapshot_0.pth', 'best_snapshot_1.pth'}}
    entries = value.get('required_files')
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError('Exactly three recovery inputs required')
    roles = set()
    for item in entries:
        role, relative = item.get('role'), item.get('relative_path')
        safe_relative(relative)
        if (role not in expected or role in roles
                or str(PurePosixPath(relative).parent) != 'checkpoints/GSE284047/' + run_id
                or PurePosixPath(relative).name not in expected[role]
                or type(item.get('bytes')) is not int or not 0 < item['bytes'] <= MAX_CORE_BYTES
                or not SHA_RE.fullmatch(item.get('sha256', ''))):
            raise ValueError('Invalid recovery input role/path/size/hash')
        roles.add(role)
    if roles != set(expected):
        raise ValueError('Incomplete recovery roles')
    return value

def capture_recovery(fetch, run_id, destination, *, attempts=3):
    """fetch(relative, byte_limit) returns exact bytes; no torch deserialization.

    Each attempt lives separately. SUCCESS requires all three declared hashes
    and identical manifest bytes before/after capture. No stale partial file is
    represented as verified; failure records contain error classes only.
    """
    validate_run(run_id)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    relative = 'checkpoints/GSE284047/' + run_id + '/checkpoint_restore_manifest.json'
    events = []
    for attempt in range(1, attempts + 1):
        folder = destination / ('attempt_' + str(attempt))
        folder.mkdir()
        try:
            before = fetch(relative, 2 * 1024**2)
            manifest = validate_restore(before, run_id)
            records = []
            for entry in manifest['required_files']:
                data = fetch(entry['relative_path'], entry['bytes'])
                if byte_record(data) != {'bytes': entry['bytes'], 'sha256': entry['sha256']}:
                    raise ValueError('Recovery input bytes/hash mismatch')
                write_new(safe_target(folder, entry['relative_path']), data)
                records.append(dict(entry))
            if fetch(relative, 2 * 1024**2) != before:
                raise ValueError('Recovery manifest changed during capture')
            write_new(safe_target(folder, relative), before)
            result = {'schema': 'dynavelo_verified_epoch_capture_v1', 'status': 'PASS_CORE_BYTES_SHA_STABLE_MANIFEST',
                      'run_id': run_id, 'completed_epoch': manifest['completed_epoch'], 'created_utc': utcnow(),
                      'verified_directory': str(folder), 'files': records + [{'role': 'restore_manifest',
                          'relative_path': relative, **byte_record(before)}], 'attempts': events + [{'attempt': attempt, 'status': 'PASS'}],
                      'scope': 'Exact-byte recovery inputs only. Trainer still verifies checkpoint contents/config on resume. Logs and predictions are separate snapshots.',
                      'durability': 'Only the locally written, verified files survive remote VM loss.'}
            write_new(destination / 'capture_result.json', json_bytes(result))
            return result
        except Exception as error:
            events.append({'attempt': attempt, 'status': 'FAIL', 'error_type': type(error).__name__})
    result = {'schema': 'dynavelo_verified_epoch_capture_v1', 'status': 'HOLD_NO_VERIFIED_RECOVERY_CAPTURE',
              'run_id': run_id, 'created_utc': utcnow(), 'attempts': events, 'files': []}
    write_new(destination / 'capture_result.json', json_bytes(result))
    return result
