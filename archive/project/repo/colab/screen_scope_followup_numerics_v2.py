"""Scoped follow-up arithmetic adapted from frozen v1, unchanged numerical core."""
import csv,io
import numpy as np
import screen_dispatch as base
import screen_followup_dispatch as old
import screen_scope_dispatch_v2 as primary
from screen_plan import seal
from dynavelo_streaming_core import EdgeAccumulator

def checked_jacobians(plan,task,session):
    return old.checked_jacobians(plan['context_plan'],task,session)

def regulator_name(plan,task):return old.regulator_name(plan['context_plan'],task)
def cells_for(plan,frame,task):return task['sample_cell_ids']

def frame_weights(plan,frame,task,indices):
    if task['sample_id']=='full_effect_population':weights=np.full(len(indices),1/len(indices),dtype=np.float64)
    else:
        rows=frame['samples'][task['sample_id']]['cells']
        if [r['cell_id'] for r in rows]!=task['sample_cell_ids']:raise ValueError('Local weight axes differ')
        weights=np.asarray([r['cell_weight'] for r in rows],dtype=np.float64)
    if [plan['context_plan']['spec']['cells'][i]['cell_id'] for i in indices]!=task['sample_cell_ids']:
        raise ValueError('Exact task/population axes differ')
    if abs(weights.sum()-1)>1e-12:raise ValueError('Weights do not sum to one')
    return weights

def group_definitions(plan,frame,task,indices):
    cells=plan['context_plan']['spec']['cells'];labels=np.array([cells[i]['label'] for i in indices],dtype=object)
    full_labels=np.array([c['label'] for c in cells],dtype=object);eligible_ids=set(frame['effect_eligible_cell_ids'])
    eligible=np.array([c['cell_id'] in eligible_ids for c in cells])
    windows=np.array(frame['full_context_frame']['global_time_windows'][str(task['seed'])]['window_index'])
    groups=[('effect_population','all',np.ones(len(indices),dtype=bool),int(eligible.sum()))]
    groups += [('label',label,labels==label,frame['effect_state_counts'][label]) for label in plan['scope']['target_labels']]
    groups += [('context_branch_intersect_effect',name,np.isin(labels,states),int((np.isin(full_labels,states)&eligible).sum()))
               for name,states in plan['context_plan']['policy']['branches'].items()]
    groups += [('full_context_time_window',str(w),windows[indices]==w,int(((windows==w)&eligible).sum())) for w in range(5)]
    return groups

def grn_task(plan,frame,task,session,writer,authorization,dependency):
    groups=group_definitions(plan,frame,task,session.indices);weights=frame_weights(plan,frame,task,session.indices)
    axes={'vx':list(session.population.genes),'vy':[r['feature_id'] for r in plan['context_plan']['spec']['motifs']]}
    accum={key:EdgeAccumulator(len(groups),len(names)) for key,names in axes.items()}
    weighted={key:{name:np.zeros((len(groups),len(names)),dtype=np.float64) for name in ('signed_sum','absolute_sum','finite_mass','nonfinite_mass')}
              for key,names in axes.items()}
    for block in checked_jacobians(plan,task,session):
        channel=block['channel'];left=block['target_start'];right=block['target_stop'];s,e=block['start'],block['stop'];values=block['values']
        for group,(_,_,mask,_) in enumerate(groups):
            select=mask[s:e]
            if not select.any():continue
            v=values[select];w=weights[s:e][select,None];finite=np.isfinite(v);valid=np.where(finite,v,0)
            accum[channel].update(v,np.full(len(v),group,dtype=np.int64),left)
            for key,part in [('signed_sum',(valid*w).sum(axis=0)),('absolute_sum',(np.abs(valid)*w).sum(axis=0)),
                             ('finite_mass',(finite*w).sum(axis=0)),('nonfinite_mass',((~finite)*w).sum(axis=0))]:
                weighted[channel][key][group,left:right]+=part
    group_rows=[{'kind':kind,'name':name,'sample_n':int(mask.sum()),'effect_frame_domain_n':n,
                 'sample_weight_sum':float(weights[mask].sum())} for kind,name,mask,n in groups]
    writer.put('edge_axes.json',{'rna_output_axis':axes['vx'],'motif_output_axis':axes['vy'],
        'motif_output_tf_mapping':list(session.population.motif_tfs),'groups':group_rows,
        'input_space':task['input_space'],'input_feature_id':task['feature_id'],'mapped_rna_feature':regulator_name(plan,task),
        'unit_warning':'RNA and motif input units are distinct; no pooled network score',
        'weighted_mean_recipe':'weighted_signed_sum / weighted_finite_mass; null when mass=0',
        'unweighted_mean_recipe':'sum / count; null when count=0; failures/nonfinite retained'})
    nonfinite=0
    for channel,a in accum.items():
        fields={key:getattr(a,key) for key in ('sum','abs_sum','count','positive','negative','zero','nonfinite')}
        nonfinite+=int(a.nonfinite[0].sum())
        fields.update({'weighted_'+key:value for key,value in weighted[channel].items()})
        np.savez(writer.directory/(channel+'_edges.npz'),**fields)
        # Effect-population count proves every target includes all sampled cells, including nonfinite entries.
        if not np.all(a.count[0]+a.nonfinite[0]==task['n_cells']):raise ValueError('Incomplete target denominator')
    writer.put('derivative_integrity.json',{'nonfinite_entries_global':nonfinite,'all_targets_retained':True})
    if nonfinite:raise ValueError('Nonfinite derivative entries retained in failed attempt; COMPLETE prohibited')
    result=primary.result(task,authorization);result.update(input_space=task['input_space'],feature_id=task['feature_id'],
        tf_stable_id=task['tf_stable_id'],epsilon=task['epsilon'],matched_baseline_dependency=dependency,
        groups=len(groups),rna_targets=len(axes['vx']),motif_targets=len(axes['vy']),
        interpretation='Local one-sided full-model input sensitivity, not direct binding or causal GRN')
    return result

def epsilon_task(plan,frame,task,session,writer,authorization,dependency):
    arrays={c:np.lib.format.open_memmap(writer.directory/(c+'_derivative.npy'),mode='w+',dtype=np.float64,
             shape=(task['n_cells'],width)) for c,width in [('vx',session.model.x_dim),('vy',session.model.y_dim)]}
    nonfinite=0
    try:
        for block in checked_jacobians(plan,task,session):
            arrays[block['channel']][block['start']:block['stop'],block['target_start']:block['target_stop']]=block['values']
            nonfinite+=int((~np.isfinite(block['values'])).sum())
        for value in arrays.values():value.flush()
    finally:
        for value in arrays.values():value._mmap.close()
    writer.put('diagnostic_axes.json',{'cell_ids':cells_for(plan,frame,task),'rna_targets':list(session.population.genes),
        'motif_targets':[r['feature_id'] for r in plan['context_plan']['spec']['motifs']],'nonfinite_entries':nonfinite})
    if nonfinite:raise ValueError('Nonfinite epsilon derivatives retained; COMPLETE prohibited')
    result=primary.result(task,authorization);result.update(feature_id=task['feature_id'],input_space=task['input_space'],
        epsilon=task['epsilon'],matched_baseline_dependency=dependency,selection='Frozen lexical IDs, no effect-dependent selection')
    return result

def full_cell_task(plan,frame,task,session,writer,authorization,dependency):
    if task['mode'] not in {'rna-only','official-joint'}:raise ValueError('Unsupported full-cell mode')
    ids=cells_for(plan,frame,task);weights=frame_weights(plan,frame,task,session.indices)
    h=[session.population.gene_index[name] for name in plan['context_plan']['policy']['primary']['H']]
    scale=frame['full_context_frame']['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid fixed RMS')
    values={k:np.full(len(ids),np.nan) for k in ('projection_H','delta_time_raw','delta_vx_norm','delta_vy_norm','delta_vz_norm')}
    expected=iter(session.batches)
    for block in session.perturbations([task['feature_id']],mode=task['mode']):
        start,stop=block['start'],block['stop']
        if (next(expected,None)!=(start,stop) or block['regulator']!=task['feature_id'] or block['mode']!=task['mode']
            or block['motif_changed']!=(task['mode']=='official-joint')
            or not np.array_equal(block['population_indices'],session.indices[start:stop])):raise ValueError('Full-cell block/mode mismatch')
        baseline=np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h];norm=np.linalg.norm(baseline,axis=1)
        if not np.isfinite(norm).all() or np.any(norm<1e-8*scale) or np.any(norm<=0):raise ValueError('No candidate-specific direction remasking')
        values['projection_H'][start:stop]=np.sum(block['delta']['vx'][:,h]*(baseline/norm[:,None]),axis=1)/scale
        values['delta_time_raw'][start:stop]=block['delta']['time']
        for c in ('vx','vy','vz'):values['delta_'+c+'_norm'][start:stop]=np.linalg.norm(block['delta'][c],axis=1)
    if next(expected,None) is not None or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Incomplete/nonfinite full-cell effects')
    result=primary.result(task,authorization);result.update(primary.summarize(plan,frame,task,values,weights,session.indices))
    result.update(feature_id=task['feature_id'],tf_stable_id=task['tf_stable_id'],matched_baseline_dependency=dependency,
        H=plan['context_plan']['policy']['primary']['H'],full_context_common_RMS=scale,shortlist_sha256=task['shortlist_sha256'],
        interpretation='Locked top20 on fixed target-common population; full-context calibration; no full-TF full-cell ranking or rescue claim')
    stream=io.StringIO(newline='');table=csv.writer(stream,delimiter='\t')
    table.writerow(['cell_id','population_index','label','cell_weight','full_original_window']+list(values))
    windows=frame['full_context_frame']['global_time_windows'][str(task['seed'])]['window_index']
    for i,position in enumerate(session.indices):
        table.writerow([ids[i],int(position),plan['context_plan']['spec']['cells'][position]['label'],float(weights[i]),windows[position]]+[float(v[i]) for v in values.values()])
    writer.put('cell_effects.tsv',stream.getvalue().encode('utf-8'));return result

def _epsilon_comparison(registry,guard=None):
    old.checkpoint(guard)
    coverage=registry.coverage(checkpoint=guard.check if guard else None)
    old.checkpoint(guard)
    if not coverage['all_expected_tasks_complete']:raise PermissionError('Complete diagnostic family required')
    rows=[]
    for task in registry.book['tasks']:
        old.checkpoint(guard)
        reference=[t for t in registry.book['tasks'] if all(t[k]==task[k] for k in ('seed','feature_id','input_space')) and t['epsilon']==1e-4]
        if len(reference)!=1:raise ValueError('Diagnostic reference must be unique')
        d,_=base.completed_directory(registry,task);ref,_=base.completed_directory(registry,reference[0])
        for channel in ('vx','vy'):
            a=np.load(d/(channel+'_derivative.npy'),mmap_mode='r',allow_pickle=False)
            b=np.load(ref/(channel+'_derivative.npy'),mmap_mode='r',allow_pickle=False)
            error2=ref2=0.;same=total=0
            if a.shape!=b.shape or a.dtype!=np.float64 or b.dtype!=np.float64:raise ValueError('Diagnostic arrays differ')
            for start in range(0,len(a),256):
                old.checkpoint(guard)
                x=a[start:start+256];y=b[start:start+256]
                if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite diagnostic payload')
                error2+=float(np.sum((x-y)**2,dtype=np.float64));ref2+=float(np.sum(y*y,dtype=np.float64))
                same+=int((np.sign(x)==np.sign(y)).sum());total+=x.size
            if not np.isfinite(error2+ref2):raise ValueError('Diagnostic aggregate overflow')
            rows.append({'seed':task['seed'],'feature_id':task['feature_id'],'input_space':task['input_space'],
                'epsilon':task['epsilon'],'reference_epsilon':1e-4,'channel':channel,'n_entries':total,
                'absolute_L2':float(np.sqrt(error2)),'reference_L2':float(np.sqrt(ref2)),
                'relative_L2':float(np.sqrt(error2/ref2)) if ref2>0 else None,'sign_agreement':same/total})
            a._mmap.close();b._mmap.close()
    return seal({'schema':'dynavelo_scope_epsilon_comparison_v2','scope_plan_sha256':registry.book['scope_plan_sha256'],
        'scope_sha256':registry.book['scope_sha256'],'scope_frame_sha256':registry.book['scope_frame_sha256'],
        'shortlist_sha256':registry.book['shortlist_sha256'],
        'book_sha256':registry.book['book_sha256'],'coverage_sha256':coverage['coverage_sha256'],'rows':rows,
        'automatic_stability_PASS':False,'epsilon_selected':None,'claim_scope':'engineering numerical sensitivity description only'},'summary_sha256')

def epsilon_comparison(registry,guard=None):
    with registry.dependency_audit(guard.check if guard else None):
        return _epsilon_comparison(registry,guard)

