"""Read-only audit of frozen score selections and existing seed matrices.
No model execution, derivative calculation or differential-expression fitting.
Requires numpy and pandas. Inputs must exist; output directory must be new.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--score-file',type=Path,required=True)
parser.add_argument('--selection-file',type=Path,required=True)
parser.add_argument('--matrix-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
scores=args.score_file.resolve();selected=args.selection_file.resolve()
INPUT=args.matrix_dir.resolve();OUT=args.output_dir.resolve()
assert not OUT.exists(), 'Choose a new output directory; inputs and previous audits are protected.'
assert OUT!=INPUT and INPUT not in OUT.parents
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
files=[scores,selected,INPUT/'axes.json']+[INPUT/f'weighted_signed_mean_seed{s}.npy' for s in range(3)]
before={str(p):sha(p) for p in files}
(OUT/'Source_Data').mkdir(parents=True)
(OUT/'Quality_Checks').mkdir()
df=pd.read_csv(scores,sep='\t',float_precision='round_trip')
arch=pd.read_csv(selected,sep='\t',float_precision='round_trip')
axes=json.loads((INPUT/'axes.json').read_text(encoding='utf-8'))
inp={x['gene']:x['index'] for x in axes['inputs']}
target={x['index']:x['gene'] for x in axes['targets']}
arrays=[np.load(INPUT/f'weighted_signed_mean_seed{s}.npy',mmap_mode='r',allow_pickle=False) for s in range(3)]
assert all(a.shape==(1191,3006) for a in arrays)
assert df.groupby('candidate_gene').size().eq(3005).all()
rows=[];edges=[];checks=[]
variants=[c for c in df if c.startswith(('B0_','B1_','B2_'))]
for variant in variants:
    for gene,g in df.groupby('candidate_gene',sort=True):
        g=g[(g.target_gene!=gene)&np.isfinite(g[variant])&(g[variant]>0)].sort_values([variant,'target_index'],ascending=[False,True],kind='stable')
        threshold=float(g.iloc[49][variant]);exact=g.iloc[:50];inclusive=g[g[variant]>=threshold]
        for mode,chosen in [('exact50',exact),('boundary_inclusive',inclusive)]:
            ids=chosen.target_index.astype(int).to_numpy()
            assert all(target[j]==v for j,v in zip(ids,chosen.target_gene))
            values=np.stack([a[inp[gene],ids] for a in arrays],axis=1)
            positive=(values>0).all(axis=1);negative=(values<0).all(axis=1)
            finite=np.isfinite(values).all(axis=1)
            stable=(positive|negative)&finite
            # Independent scalar implementation cross-checks the vectorized predicate.
            scalar=[all(v>0 for v in vals) or all(v<0 for v in vals) for vals in values.tolist()]
            assert stable.tolist()==scalar and finite.all()
            archived=arch[(arch.variant==variant)&(arch.candidate_gene==gene)]
            if mode=='exact50':archived=archived[archived.in_exact50.astype(str).str.lower().eq('true')]
            else:archived=archived[archived.in_boundary_inclusive.astype(str).str.lower().eq('true')]
            assert set(ids)==set(archived.target_index.astype(int))
            rows.append(dict(score=variant[:2],selection=mode,candidate=gene,n_edges=len(ids),all_positive=int(positive.sum()),all_negative=int(negative.sum()),same_sign=int(stable.sum()),mixed_or_zero=int((~stable).sum()),zero_present=int((values==0).any(axis=1).sum()),nonfinite=int((~finite).sum()),same_sign_fraction=float(stable.mean()),boundary_score=threshold))
            for rank,((_,r),vals,ok) in enumerate(zip(chosen.iterrows(),values,stable),1):
                edges.append(dict(score=variant[:2],selection=mode,candidate=gene,target=r.target_gene,target_index=int(r.target_index),selection_rank=rank,score_value=r[variant],seed0=vals[0],seed1=vals[1],seed2=vals[2],strict_same_sign=bool(ok)))
            checks.append(f'{variant[:2]} {gene} {mode}: archived edge identities match')
summary=pd.DataFrame(rows)
pooled=summary.groupby(['score','selection'],sort=True)[['n_edges','all_positive','all_negative','same_sign','mixed_or_zero','zero_present','nonfinite']].sum().reset_index()
pooled['same_sign_fraction']=pooled.same_sign/pooled.n_edges
data=OUT/'Source_Data'
summary.to_csv(data/'S6_Direction_stability_by_candidate.csv',index=False)
pooled.to_csv(data/'S6_Direction_stability_pooled.csv',index=False)
pd.DataFrame(edges).to_csv(data/'S6_Selected_edges_with_seed_signs.tsv',sep='\t',index=False)
pairs=[]
for gene,t in [('CEBPA','MPO'),('CEBPA','CSF3R'),('XBP1','HSPA5')]:
    j=next(k for k,v in target.items() if v==t)
    vals=[float(a[inp[gene],j]) for a in arrays]
    pairs.append(dict(regulator=gene,target=t,evaluation_cells=192,seed0=vals[0],seed1=vals[1],seed2=vals[2],positive_seeds=sum(v>0 for v in vals),negative_seeds=sum(v<0 for v in vals)))
pd.DataFrame(pairs).to_csv(data/'S3_Primary_matrix_three_seed_signs.csv',index=False)
after={p:sha(Path(p)) for p in before}
assert before==after
report=dict(input_integrity='all protected inputs unchanged',protected_file_count=len(before),strict_definition='All three finite seed values > 0 OR all three < 0; no tolerance; zeros are not same-sign.',selection='Positive non-self frozen score, descending; ties resolved by original target index ascending. Inclusive selection includes every score >= the 50th score.',matrix_shape=[1191,3006],evaluation_cells=192,checks=checks,input_sha256=before,pooled=pooled.to_dict('records'))
(OUT/'Quality_Checks/S6_READ_ONLY_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(pooled.to_string(index=False))
