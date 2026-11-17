"""Primary runtime scope contract: pure planning, no execution authority.

An intact v1 plan/frame supplies training context and calibration. A separate
v2 effect frame supplies the estimand and samples. New schemas deliberately
reject v1 execution. Stage authorization separately verifies files and reviews.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from screen_plan import (canonical, identity, seal, check_seal, validate_plan,
                         freeze_baseline_frame, sample_cells, SHA)
from screen_task_registry import batches

TARGET_LABELS = ('HSC', 'Prog DC', 'DC')
SAMPLE_SEEDS = {'primary': 73011, 'stability': 73012}
SUPPORTED_STAGES = ('baseline', 'matched-baseline', 'screen')


def clone(value):
    return json.loads(canonical(value))


def source_bindings():
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('screen_scope_plan_runtime_v2.py', 'screen_plan.py', 'screen_task_registry.py')}


def build_scope_plan(context_plan):
    validate_plan(context_plan)
    purpose=context_plan['spec']['purpose']
    labels = {row['label'] for row in context_plan['spec']['cells']}
    if not set(TARGET_LABELS) <= labels:
        raise ValueError('All three fixed target states required in trained context')
    spec = context_plan['spec']
    target = [row['cell_id'] for row in spec['cells'] if row['label'] in TARGET_LABELS]
    if purpose=='human_exploratory_input_sensitivity':
        counts={label:sum(c['label']==label for c in spec['cells']) for label in TARGET_LABELS}
        if (len(spec['cells'])!=6035 or len(spec['genes'])!=3006 or len(spec['motifs'])!=365
            or len(context_plan['tf_universe'])!=1191 or counts!={'HSC':1307,'Prog DC':402,'DC':175}):
            raise ValueError('Human context/target/feature/TF axes differ from adopted scope')
    scope = {'schema': 'dynavelo_effect_scope_v2',
             'trained_context': 'entire unchanged context_plan cell/feature/label/split axis',
             'target_labels': list(TARGET_LABELS), 'target_cell_ids_before_baseline': target,
             'target_cell_axis_sha256': identity(target),
             'eligible_effect_population': 'fixed target IDs intersect full-context common baseline-evaluable IDs',
             'calibration_population': 'unchanged full-context common baseline-evaluable IDs across seeds 0,1,2',
             'floor_H_time': 'full context minima and H; per-seed rank/windows on entire original axis',
             'weights': 'eligible target N_label/N; selected cell weight N_label/N/n_label',
             'empty_target_state': 'HOLD; no execution after dropping an empty state',
             'preexisting_context_HOLD': 'retain including empty non-target state; no relaxation',
             'samples': {'seeds': SAMPLE_SEEDS, 'per_state_cap': 64,
                         'algorithm': context_plan['policy']['sampling']['algorithm']}}
    result = {'schema': 'dynavelo_scope_plan_v2', 'purpose': purpose,
              'context_plan': clone(context_plan), 'context_plan_sha256': context_plan['plan_sha256'],
              'scope': scope, 'scope_sha256': identity(scope),
              'context_invariants': {'input_identity_sha256': context_plan['input_identity_sha256'],
                  'axes_sha256': context_plan['axes_sha256'], 'global_minima_sha256': identity(spec['global_minima']),
                  'H_sha256': identity(context_plan['policy']['primary']['H']),
                  'checkpoint_set_sha256': context_plan['checkpoint_set_sha256'],
                  'tf_universe_sha256': identity(context_plan['tf_universe']),
                  'full_tf_denominator': len(context_plan['tf_universe']),
                  'unmodified_policy_sha256': identity(context_plan['policy'])},
              'stability_policy': clone(context_plan['policy']['stability']),
              'human_contract_not_instantiated': {'trained_context_n': 6035, 'original_target_n': 1884,
                  'target_counts': {'HSC': 1307, 'Prog DC': 402, 'DC': 175},
                  'full_eligible_tf_denominator': 1191, 'model_seeds': [0, 1, 2]},
              'source_sha256': source_bindings(), 'execution_authorized': False,
              'scientific_PASS': False, 'production_dispatch_supported': 'primary_only_requires_bound_review',
              'status': 'PLAN_ONLY_STAGE_AUTHORIZATION_REQUIRED'}
    return seal(result, 'scope_plan_sha256')


def validate_scope_plan(plan):
    check_seal(plan, 'scope_plan_sha256')
    if canonical(build_scope_plan(plan['context_plan'])) != canonical(plan):
        raise ValueError('Scope fails intact context/fixed scope/source reproduction')
    return plan


def _derive_frame(plan, summaries):
    context = plan['context_plan']
    original = context['spec']['cells']
    # Full-context fixed point and RMS precede target intersection, unchanged.
    calibration = freeze_baseline_frame(context, summaries)
    common = set(calibration['common_eligible_cell_ids'])
    target = set(plan['scope']['target_cell_ids_before_baseline'])
    eligible = [row['cell_id'] for row in original if row['cell_id'] in common & target]
    eligible_set = set(eligible)
    counts = {label: sum(row['label'] == label and row['cell_id'] in eligible_set
                         for row in original) for label in TARGET_LABELS}
    empty = [label for label in TARGET_LABELS if counts[label] == 0]
    reasons = []
    if calibration['status'] != 'READY_FOR_SEPARATE_FRAME_REVIEW':
        reasons.append('CONTEXT_' + calibration['status'])
    if empty:
        reasons.append('EMPTY_TARGET_STATE')
    samples = {name: sample_cells(original, eligible, seed) for name, seed in SAMPLE_SEEDS.items()} if eligible else {}
    return {'schema': 'dynavelo_scope_frame_v2', 'scope_plan_sha256': plan['scope_plan_sha256'],
            'scope_sha256': plan['scope_sha256'], 'context_plan_sha256': context['plan_sha256'],
            'baseline_summaries': clone(summaries), 'full_context_frame': calibration,
            'full_context_frame_sha256': calibration['frame_sha256'],
            'calibration_sha256': identity({'context_frame_sha256': calibration['frame_sha256'],
                                           'global_time_windows': calibration['global_time_windows']}),
            'effect_eligible_cell_ids': eligible, 'effect_cell_axis_sha256': identity(eligible),
            'original_target_n': len(target), 'effect_population_n': len(eligible),
            'excluded_target_cell_ids': [row['cell_id'] for row in original if row['cell_id'] in target-common],
            'effect_state_counts': counts, 'empty_target_states': empty,
            'samples': samples, 'HOLD_reasons': reasons,
            'status': 'HOLD' if reasons else 'READY_FOR_SEPARATE_SCOPE_FRAME_REVIEW',
            'execution_authorized': False, 'scientific_PASS': False}


def freeze_scope_frame(plan, summaries):
    validate_scope_plan(plan)
    return seal(_derive_frame(plan, summaries), 'scope_frame_sha256')


def validate_scope_frame(plan, frame):
    validate_scope_plan(plan)
    check_seal(frame, 'scope_frame_sha256')
    if canonical(seal(_derive_frame(plan, frame['baseline_summaries']), 'scope_frame_sha256')) != canonical(frame):
        raise ValueError('Local frame/calibration/samples fail deterministic reproduction')
    return frame


def _sample_definitions(plan, frame, stage):
    original = plan['context_plan']['spec']['cells']
    if stage == 'baseline':
        return {'full_trained_context': {'ids': [row['cell_id'] for row in original], 'weights': None}}
    selected = {name: {'ids': [row['cell_id'] for row in sample['cells']],
                       'weights': [row['cell_weight'] for row in sample['cells']]}
                for name, sample in frame['samples'].items()}
    if stage == 'matched-baseline':
        n = frame['effect_population_n']
        selected['full_effect_population'] = {'ids': frame['effect_eligible_cell_ids'], 'weights': [1/n]*n}
    return selected


def scope_task_book(plan, stage, frame=None):
    validate_scope_plan(plan)
    if stage not in SUPPORTED_STAGES:
        raise NotImplementedError('v2 follow-up, GRN, shortlist and execution are not implemented')
    if stage == 'baseline':
        if frame is not None:
            raise ValueError('Full-context baseline tasks precede frame derivation')
    else:
        validate_scope_frame(plan, frame)
        if frame['status'] != 'READY_FOR_SEPARATE_SCOPE_FRAME_REVIEW':
            raise PermissionError('Held scope cannot produce effect/matched task descriptions')
    context = plan['context_plan']
    geometry = context['spec']['chunk_geometry']
    samples = _sample_definitions(plan, frame, stage)
    tasks = []
    for model in sorted(context['spec']['models'], key=lambda item: item['seed']):
        for name, sample in samples.items():
            regulators = context['tf_universe'] if stage == 'screen' else [{'stable_id': '__baseline__', 'feature_id': '__baseline__'}]
            for regulator in regulators:
                body = {'schema': 'dynavelo_scope_task_v2', 'scope_plan_sha256': plan['scope_plan_sha256'],
                        'scope_sha256': plan['scope_sha256'], 'context_plan_sha256': context['plan_sha256'],
                        'scope_frame_sha256': frame['scope_frame_sha256'] if frame else None,
                        'full_context_frame_sha256': frame['full_context_frame_sha256'] if frame else None,
                        'calibration_sha256': frame['calibration_sha256'] if frame else None,
                        'context_invariants_sha256': identity(plan['context_invariants']),
                        'checkpoint_sha256': model['checkpoint']['sha256'], 'seed': model['seed'],
                        'stage': stage, 'sample_id': name, 'sample_cell_ids': sample['ids'],
                        'sample_cell_axis_sha256': identity(sample['ids']), 'sample_weights_sha256': identity(sample['weights']),
                        'n_cells': len(sample['ids']), 'tf_stable_id': regulator['stable_id'],
                        'feature_id': regulator['feature_id'], 'mode': 'rna-only' if stage == 'screen' else 'evaluation-fixed',
                        'chunk_geometry': geometry, 'batches': batches(len(sample['ids']), geometry['cells']),
                        'execution_authorized': False}
                task = seal(body, 'task_id')
                task['assigned_shard'] = int(task['task_id'][:16], 16) % geometry.get('regulator_shards', 1)
                tasks.append(task)
    return seal({'schema': 'dynavelo_scope_task_book_v2', 'scope_plan_sha256': plan['scope_plan_sha256'],
                 'scope_sha256': plan['scope_sha256'], 'stage': stage,
                 'scope_frame_sha256': frame['scope_frame_sha256'] if frame else None,
                 'tasks': tasks, 'expected_tasks': len(tasks), 'full_tf_denominator': len(context['tf_universe']),
                 'technical_attempt_limit': 3, 'execution_authorized': False,
                 'coverage_contract': 'All 3 seeds and both samples across the intact TF universe; failures retained'}, 'book_sha256')


def validate_scope_book(plan, book, frame=None):
    check_seal(book, 'book_sha256')
    if canonical(scope_task_book(plan, book['stage'], frame)) != canonical(book):
        raise ValueError('Scope book differs from the exact complete pure task family')
    return book


def baseline_cache_binding(plan, book, task_id, *, frame=None):
    """Metadata envelope only; not permission to open/reuse any cache bytes."""
    validate_scope_book(plan, book, frame)
    selected = [task for task in book['tasks'] if task['task_id'] == task_id]
    if len(selected) != 1 or selected[0]['stage'] not in ('baseline', 'matched-baseline'):
        raise ValueError('Unique planned baseline task required')
    task = selected[0]
    return {'schema': 'dynavelo_scoped_baseline_binding_v2', 'scope_plan_sha256': plan['scope_plan_sha256'],
            'scope_sha256': plan['scope_sha256'], 'scope_frame_sha256': task['scope_frame_sha256'],
            'context_plan_sha256': task['context_plan_sha256'], 'context_invariants_sha256': task['context_invariants_sha256'],
            'full_context_frame_sha256': task['full_context_frame_sha256'], 'calibration_sha256': task['calibration_sha256'],
            'task_id': task_id, 'book_sha256': book['book_sha256'], 'checkpoint_sha256': task['checkpoint_sha256'],
            'sample_cell_axis_sha256': task['sample_cell_axis_sha256'], 'sample_weights_sha256': task['sample_weights_sha256'],
            'batches': task['batches'], 'n_cells': task['n_cells'], 'execution_authorized': False}


def seal_cache_envelope(plan, book, task_id, core_manifest_sha256, *, frame=None):
    if not isinstance(core_manifest_sha256, str) or not SHA.fullmatch(core_manifest_sha256):
        raise ValueError('Core manifest hash required even for the synthetic envelope')
    return seal({'schema': 'dynavelo_scoped_baseline_envelope_v2',
                 'binding': baseline_cache_binding(plan, book, task_id, frame=frame),
                 'core_manifest_sha256': core_manifest_sha256,
                 'payload_verification': 'required_by_dispatcher_COMMIT_and_core_hash_checks'}, 'cache_identity_sha256')


def validate_cache_envelope(plan, book, task_id, envelope, expected_core_manifest_sha256, *, frame=None):
    check_seal(envelope, 'cache_identity_sha256')
    expected = seal_cache_envelope(plan, book, task_id, expected_core_manifest_sha256, frame=frame)
    if canonical(expected) != canonical(envelope):
        raise ValueError('Cache scope, calibration, sample, checkpoint, geometry or manifest differs')
    return envelope


def authorize_execution(*args, **kwargs):
    raise PermissionError('Use separately verified stage authorization; a scope plan grants no execution')
