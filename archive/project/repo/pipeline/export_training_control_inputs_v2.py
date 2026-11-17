"""V2 V/M exporter preserves verified author-coded subject; default plan-only."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
ROOT=Path('D:/dyc_7/DynaVelo_Research') if os.name=='nt' else Path('/mnt/d/dyc_7/DynaVelo_Research')
FACTUAL={'rna':{'relative_path':'data/processed/colab_minimal/HSPC/adata_rna.h5ad','bytes':31041196,'sha256':'245ce8a44dc5b1b9d1835e4f78ef604afb291bfc1841dff1a1777929b3a9a8fe'},'atac':{'relative_path':'data/processed/colab_minimal/HSPC/adata_atac.h5ad','bytes':9709799,'sha256':'968aa6c255b3c45ba1ff6103219a822dd7319aa14e5d9a7a3c50fedc39c3aa29'}}
MANIFEST_SHA='89aa75e78d34ea72b2d0411977f484ec40e7a580dfe938859fd14a09d1e4752a'
FOUNDATION='repo/pipeline/training_control_permutation.py'
FOUNDATION_SHA='02a33793ad71433b838f1565e1f2ff6c568ab04a15d6ba6e3366f6de5c13b072'
PROPOSAL='results/reports/training_negative_controls_proposal_v1.1.0.json'
PROPOSAL_SHA='2c700be12411fd4bcbf7b4db2e6101f27f227bf0af375d7ad6327de1c3123bf3'
CODED_EVIDENCE='results/reports/GSE284047_GSE274113_reuse_identity_audit.json'
CODED_EVIDENCE_SHA='8eacced56b17321dfa88c324a77902dd072ad959a96f6b9e47a35775459ec0b8'
CODED_CONTEXT={'schema':'dynavelo_author_coded_subject_v1','library_id':'8489-MV-1','author_subject_code':'RO03744','dbgap_study':'phs002915','gap_subject_id':'5343494','author_coded_unit_count':1,'personal_identity':'unknown','external_person_overlap':'unknown','verified_donor_id_field_semantics':'Null denotes unresolved personal identity linkage; author coded subject is known.','evidence':{'relative_path':CODED_EVIDENCE,'sha256':CODED_EVIDENCE_SHA}}
SEEDS={'V':92100,'M':93100};SCOPE='across-author-labels';SHA_RE=re.compile('[0-9a-f]{64}\\Z')
ASSEMBLY='uns/model_input_assembly_json';CONTROL='uns/dynavelo_training_control_json'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def safe(root,rel):
 if not isinstance(rel,str) or not rel or '\\' in rel or ':' in rel or any(x in {'','.','..'} for x in rel.split('/')) or Path(rel).is_absolute():raise ValueError('Unsafe relative path')
 root=Path(root).absolute();p=root/rel
 if root.resolve()!=root or p.resolve()!=p or not p.is_relative_to(root):raise ValueError('Redirected/outside root')
 return p

def validate_metadata(meta,plan,*,fixture=False):
 if meta.get('coded_subject_provenance')!=plan.get('coded_subject_provenance') or meta.get('coded_subject_provenance')!=CODED_CONTEXT:raise ValueError('Metadata must preserve known author coded subject')
 if meta.get('schema')!='dynavelo_control_metadata_v2' or meta.get('source_inputs')!={k:v['sha256'] for k,v in plan['inputs'].items()}:raise ValueError('Metadata source differs')
 rows=meta.get('rows');fields={'cell_id','library_id','verified_donor_id','split','author_label'}
 if not isinstance(rows,list) or len(rows)!=plan['dimensions']['n_cells']:raise ValueError('Complete metadata required')
 for row in rows:
  if set(row)!=fields or any(not isinstance(row[k],str) or not row[k] for k in ['cell_id','library_id']):raise ValueError('Exact cell/library fields required')
  if row['split'] not in ('train','validation'):raise ValueError('Unknown split')
  if any(row[k] is not None and (not isinstance(row[k],str) or not row[k]) for k in ['verified_donor_id','author_label']):raise ValueError('Unknown donor/label must be null')
  if fixture and not row['cell_id'].startswith('fixture::'):raise ValueError('Synthetic IDs required')
  if not fixture and (row['library_id']!='8489-MV-1' or row['verified_donor_id'] is not None):raise ValueError('Factual library/null personal-identity field differs')
 if len({r['cell_id'] for r in rows})!=len(rows):raise ValueError('Duplicate cell identity')
 return sorted(rows,key=lambda r:r['cell_id'])

def expected_mapping(meta,plan,*,fixture=False):
 """Reconstruct a supplied mapping for verification; CLI has no map output mode."""
 import numpy as np
 rows=validate_metadata(meta,plan,fixture=fixture);groups=defaultdict(list)
 for row in rows:groups[(row['library_id'],row['verified_donor_id'],row['split'])].append(row)
 rng=np.random.default_rng(SEEDS[plan['control']]);pairs=[];audits=[];transition=defaultdict(int)
 for key in sorted(groups,key=canonical):
  members=groups[key];order=rng.permutation(len(members));known=cross=0
  for i,j in enumerate(order):
   recipient=members[int(j)];source=members[int(order[(i+1)%len(order)])]
   pairs.append({'recipient_cell_id':recipient['cell_id'],'source_cell_id':source['cell_id']})
   a,b=recipient['author_label'],source['author_label'];transition[(a,b)]+=1
   known+=int(a is not None and b is not None);cross+=int(a is not None and b is not None and a!=b)
  audits.append({'library_id':key[0],'verified_donor_id':key[1],'split':key[2],'n_cells':len(members),'singleton_unchanged':len(members)==1,'known_label_pairs':known,'cross_label_pairs':cross})
 pairs.sort(key=lambda r:r['recipient_cell_id'])
 return {'schema':'dynavelo_control_row_mapping_v2','control':plan['control'],'scope':SCOPE,'permutation_seed':SEEDS[plan['control']],
  'shared_initialization_seeds':[0,1,2],'numpy_version':np.__version__,'metadata_sha256':plan['metadata']['sha256'],
  'source_inputs':meta['source_inputs'],'coded_subject_provenance':plan['coded_subject_provenance'],'unknown_donor_field_semantics':'Null verified_donor_id count; not absence of known author coded subject.','pairs':pairs,'pairs_sha256':digest(pairs),'strata_audit':audits,
  'label_transition_counts':[{'recipient_label':k[0],'source_label':k[1],'n':v} for k,v in sorted(transition.items(),key=lambda kv:canonical(kv[0]))],
  'unknown_donor_cells':sum(r['verified_donor_id'] is None for r in rows),'unknown_donor_controlled':False,
  'resampling_to_increase_cross_label_pairs':False,'biological_validation_or_p_value':False}

def validate_mapping(mapping,meta,plan,*,fixture=False):
 expected=expected_mapping(meta,plan,fixture=fixture)
 if canonical(mapping)!=canonical(expected):raise ValueError('Mapping differs from one fixed complete within-stratum permutation')
 return expected

def value_hash(value):
 import numpy as np
 a=np.asarray(value);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(canonical(list(a.shape)))
 if a.dtype.kind in 'OUS':
  for item in a.reshape(-1):
   if isinstance(item,(str,np.str_)):raw=str(item).encode();kind=b'u'
   elif isinstance(item,(bytes,np.bytes_)):raw=bytes(item);kind=b'b'
   else:raise ValueError('Unsupported object HDF value')
   h.update(kind);h.update(len(raw).to_bytes(8,'big'));h.update(raw)
 else:h.update(a.tobytes(order='C'))
 return h.hexdigest()

def tree_hashes(handle,exclude=()):
 """Logical HDF leaf/attribute hashes; excludes only explicitly changed paths."""
 import h5py
 out={}
 def visit(name,obj):
  if any(name==x or name.startswith(x+'/') for x in exclude):return
  item={'kind':'dataset' if isinstance(obj,h5py.Dataset) else 'group','attrs':{k:value_hash(obj.attrs[k]) for k in sorted(obj.attrs)}}
  if isinstance(obj,h5py.Dataset):
   item.update(shape=list(obj.shape),dtype=str(obj.dtype));h=hashlib.sha256()
   step=262144 if obj.ndim==1 else 64
   blocks=[obj[()]] if obj.ndim==0 else (obj[i:i+step] for i in range(0,obj.shape[0],step))
   for block in blocks:h.update(value_hash(block).encode())
   item['values_sha256']=h.hexdigest()
  else:
   for key in obj:
    if not isinstance(obj.get(key,getlink=True),h5py.HardLink):raise ValueError('External/soft HDF link not allowed')
  out[name]=item
 visit('',handle);handle.visititems(visit);return out

def matrix_blocks(matrix):
 from scipy import sparse
 for i in range(0,matrix.shape[0],64):
  block=matrix[i:i+64]
  yield block.toarray() if sparse.issparse(block) else block
def matrix_equal(a,b):
 return a.shape==b.shape and str(a.dtype)==str(b.dtype) and all(value_hash(x)==value_hash(y) for x,y in zip(matrix_blocks(a),matrix_blocks(b)))
def weights(obs):
 import numpy as np
 out=np.empty(len(obs),dtype=np.float64)
 for group in ('train','validation'):
  idx=np.flatnonzero(obs['split'].astype(str).to_numpy()==group)
  labels=obs.iloc[idx]['celltype'].astype('category').cat.remove_unused_categories();counts=labels.value_counts();w=1/counts;w=w/w.sum()
  out[idx]=[w[x] for x in labels]
 return out

def inspect_inputs(rna,atac,meta,plan,*,fixture=False):
 import numpy as np
 from anndata._io.specs import read_elem
 ro,ao=read_elem(rna['obs']),read_elem(atac['obs']);rv,av=read_elem(rna['var']),read_elem(atac['var'])
 dims=plan['dimensions']
 if (len(ro),len(rv),len(av))!=(dims['n_cells'],dims['n_genes'],dims['n_motifs']) or not ro.index.is_unique or not rv.index.is_unique or not av.index.is_unique or not ro.index.equals(ao.index):raise ValueError('Paired complete axes differ')
 if list(av['TF'].astype(str))!=list(av.index.astype(str)) or not set(av.index).issubset(rv.index):raise ValueError('Motif TF axes differ')
 rows=validate_metadata(meta,plan,fixture=fixture);byid={r['cell_id']:r for r in rows}
 if set(byid)!=set(ro.index):raise ValueError('Map metadata cell axis differs')
 for obs in (ro,ao):
  if obs[['celltype','split','latent_time']].isna().any().any() or not np.isfinite(np.asarray(obs['latent_time'],dtype=float)).all():raise ValueError('Required recipient annotations absent')
  for cell,row in obs.iterrows():
   if str(row['split'])!=byid[cell]['split'] or str(row['celltype'])!=byid[cell]['author_label']:raise ValueError('Metadata disagrees with factual labels/split')
 assemblies=[]
 for handle in (rna,atac):
  assembly=json.loads(read_elem(handle[ASSEMBLY]));assemblies.append(assembly)
  if CONTROL in handle or 'training_control' in assembly or assembly.get('split_seed')!=0:raise ValueError('Stacked control/invalid original provenance')
 if assemblies[0]!=assemblies[1]:raise ValueError('Paired original assembly differs')
 if not fixture:
  order=np.random.default_rng(0).permutation(len(ro));ids=np.asarray(sorted(ro.index));expected=set(ids[order[:int(.8*len(ro))]])
  if set(ro.index[ro['split'].astype(str)=='train'])!=expected:raise ValueError('Original fixed split differs')
 return ro,ao,assemblies[0]
def write_new(p,v):
 raw=(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p)
 finally:tmp.unlink()
def record(p,root):return {'relative_path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify_record(root,rec):
 if set(rec)!={'relative_path','bytes','sha256'} or type(rec['bytes']) is not int or rec['bytes']<0 or not SHA_RE.fullmatch(rec['sha256']):raise ValueError('Bad file record')
 p=safe(root,rec['relative_path'])
 if not p.is_file() or p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError('Frozen file drift: '+rec['relative_path'])
 return p
def validate_plan(p,*,fixture=False):
 if p.get('coded_subject_provenance')!=CODED_CONTEXT or sha(ROOT/CODED_EVIDENCE)!=CODED_EVIDENCE_SHA:raise ValueError('Author coded subject or evidence differs')
 if p.get('schema')!='dynavelo_control_export_plan_v2' or p.get('control') not in SEEDS or p.get('scope')!=SCOPE:raise ValueError('Unsupported control/scope')
 if p.get('permutation_seed')!=SEEDS[p['control']] or p.get('shared_initialization_seeds')!=[0,1,2]:raise ValueError('Fixed seed/list differs')
 if p.get('foundation_sha256')!=FOUNDATION_SHA or p.get('proposal_sha256')!=PROPOSAL_SHA or p.get('exporter_sha256')!=sha(__file__):raise ValueError('Code/proposal differs')
 if any(p.get(k) is not False for k in ['scientific_pass','training_authorized','candidate_access_authorized']):raise ValueError('Export-only scope required')
 if not re.fullmatch('[A-Za-z0-9_-]{8,80}',p.get('export_id','')) or set(p.get('inputs',{}))!={'rna','atac'}:raise ValueError('Unique export ID and paired inputs required')
 if fixture:
  if p.get('source_kind')!='engineering_fixture':raise ValueError('Synthetic fixture required')
 elif (p.get('source_kind')!='factual_hspc' or p['inputs']!=FACTUAL or p.get('factual_manifest_sha256')!=MANIFEST_SHA or p.get('dimensions')!={'n_cells':6035,'n_genes':3006,'n_motifs':365} or p.get('library_id')!='8489-MV-1' or p.get('donor_status')!='coded_subject_known_personal_identity_unknown'):raise ValueError('Only frozen full HSPC inputs and coded-subject status allowed')
 for key in ['metadata','mapping']:
  rec=p.get(key,{})
  if set(rec)!={'relative_path','bytes','sha256'} or not SHA_RE.fullmatch(rec.get('sha256','')):raise ValueError('Frozen '+key+' required')
 return p

def bound_plan(root,path,expected_sha):
 path=Path(path).absolute()
 if not path.is_relative_to(root) or safe(root,path.relative_to(root).as_posix())!=path or not SHA_RE.fullmatch(expected_sha):raise ValueError('Exact root plan path/SHA required')
 with path.open('rb') as f:raw=f.read(2*1024**2+1)
 if len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Plan bytes differ from reviewed SHA')
 return json.loads(raw.decode('utf-8-sig')),raw

def export_inputs(root,plan_path,plan_sha,*,_fixture=False):
 """No model calls. Return COMPLETE only after independent written-file verification."""
 root=Path(root).absolute();plan,plan_bytes=bound_plan(root,plan_path,plan_sha);validate_plan(plan,fixture=_fixture)
 if _fixture:
  if 'tmp_cache' not in root.parts or root==ROOT:raise ValueError('Fixture root must be isolated tmp_cache')
 elif root!=ROOT:raise ValueError('Exact project root required')
 if not SHA_RE.fullmatch(plan_sha) or plan.get('execution_authorized') is not True or plan.get('reviewer')!='/root':raise PermissionError('Separate root export approval required')
 import importlib.metadata
 import h5py,numpy as np
 from anndata._io.specs import read_elem,write_elem
 versions={p:importlib.metadata.version(p) for p in ['numpy','anndata','h5py','pandas','scipy']}
 if versions!={'numpy':'1.26.4','anndata':'0.10.9','h5py':'3.12.1','pandas':'2.2.3','scipy':'1.13.1'}:raise ValueError('Frozen serialization/permutation versions differ')
 if sha(ROOT/FOUNDATION)!=FOUNDATION_SHA or sha(ROOT/PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Reviewed foundation/proposal drift')
 paths={k:verify_record(root,v) for k,v in plan['inputs'].items()}
 meta_path=verify_record(root,plan['metadata']);map_path=verify_record(root,plan['mapping']);meta=read(meta_path);mapping=read(map_path)
 validate_mapping(mapping,meta,plan,fixture=_fixture)
 out=safe(root,'data/processed/training_controls/'+plan['export_id']);out.mkdir(parents=True,exist_ok=False)
 write_new(out/'STARTED.json',{'schema':'dynavelo_control_export_started_v1','plan_sha256':plan_sha,'control':plan['control'],'utc':datetime.now(timezone.utc).isoformat(),'scientific_pass':False})
 try:
  with (out/'plan.json').open('xb') as f:f.write(plan_bytes);f.flush();os.fsync(f.fileno())
  shutil.copyfile(meta_path,out/'metadata.json');shutil.copyfile(map_path,out/'mapping.json')
  outputs={k:out/('adata_'+k+'.h5ad') for k in ['rna','atac']}
  for k in outputs:
   with outputs[k].open('xb') as dst,paths[k].open('rb') as src:shutil.copyfileobj(src,dst,4*1024**2);dst.flush();os.fsync(dst.fileno())
  with h5py.File(paths['rna'],'r') as r,h5py.File(paths['atac'],'r') as a:
   ro,ao,assembly=inspect_inputs(r,a,meta,plan,fixture=_fixture);pos={cell:i for i,cell in enumerate(ro.index)};mapping_by={p['recipient_cell_id']:p['source_cell_id'] for p in mapping['pairs']};indices=np.asarray([pos[mapping_by[c]] for c in ro.index],dtype=np.int64)
   if sorted(indices.tolist())!=list(range(len(ro))):raise ValueError('Not a full one-to-one map')
   for handle,cols in [(r,plan['dimensions']['n_genes']),(a,plan['dimensions']['n_motifs'])]:
    shape=handle['X'].shape if isinstance(handle['X'],h5py.Dataset) else tuple(handle['X'].attrs['shape'])
    if tuple(shape)!=(len(ro),cols):raise ValueError('Complete X shape differs')
   velocity=read_elem(r['layers/velocity'])
   if velocity.shape!=(len(ro),plan['dimensions']['n_genes']) or velocity.dtype.kind!='f':raise ValueError('Floating complete velocity required')
   for block in matrix_blocks(velocity):
    if np.isinf(block).any():raise ValueError('Velocity infinity forbidden')
   mask_names=[x for x in ['velocity_mask','velocity_genes_mask'] if 'layers/'+x in r]
   for name in mask_names:
    mask=read_elem(r['layers/'+name])
    if mask.shape!=velocity.shape or any(not np.array_equal(m,~np.isnan(v)) for m,v in zip(matrix_blocks(mask),matrix_blocks(velocity))):raise ValueError('Explicit mask differs from official NaN mask')
   changed={'rna':['layers/velocity']+['layers/'+x for x in mask_names],'atac':[]} if plan['control']=='V' else {'rna':[],'atac':['X']}
   lineage={'schema':'dynavelo_training_control_lineage_v2','coded_subject_provenance':plan['coded_subject_provenance'],'control':plan['control'],'scope':SCOPE,'plan_sha256':plan_sha,'mapping_sha256':plan['mapping']['sha256'],'metadata_sha256':plan['metadata']['sha256'],'factual_inputs':plan['inputs'],'permutation_seed':SEEDS[plan['control']],'shared_initialization_seeds':[0,1,2],'exporter_sha256':sha(__file__),'original_assembly_sha256':digest(assembly),'recipient_weights_unchanged':True,'weighted_modality_marginals_preserved':False,'retained_annotations':'Original factual latent_time, embeddings, graphs and confidence are retained byte-logically; not recomputed control results. Only original trainer input fields are authorized.','scientific_pass':False}
   derived=dict(assembly);derived['training_control']=lineage
   for k,source in [('rna',r),('atac',a)]:
    exclude=[ASSEMBLY,CONTROL,*changed[k]];before=tree_hashes(source,exclude)
    with h5py.File(outputs[k],'r+') as dest:
     for name in changed[k]:
      original=read_elem(source[name]);permuted=original[indices].copy()
      if name=='X' and any(not np.isfinite(x).all() for x in matrix_blocks(original)):raise ValueError('Motif X must be finite')
      parent,leaf=name.rsplit('/',1) if '/' in name else ('',name);group=dest[parent] if parent else dest
      del group[leaf];write_elem(group,leaf,permuted)
      if not matrix_equal(read_elem(dest[name]),permuted):raise ValueError('Serialized row values/dtype/NaN differ')
     del dest[ASSEMBLY];write_elem(dest['uns'],'model_input_assembly_json',canonical(derived).decode());write_elem(dest['uns'],'dynavelo_training_control_json',canonical(lineage).decode());dest.flush()
    with h5py.File(outputs[k],'r') as dest:
     if tree_hashes(dest,exclude)!=before:raise ValueError('Unapproved field/axis/annotation change')
     if json.loads(read_elem(dest[ASSEMBLY]))!=derived or json.loads(read_elem(dest[CONTROL]))!=lineage:raise ValueError('Written provenance differs')
     obs=read_elem(dest['obs'])
     if value_hash(weights(obs))!=value_hash(weights(ro if k=='rna' else ao)):raise ValueError('Recipient split weights changed')
   nan_count=sum(int(np.isnan(b).sum()) for b in matrix_blocks(velocity))
   axes={'cell_axis_sha256':digest(list(ro.index.astype(str))),'rna_axis_sha256':digest(list(read_elem(r['var']).index.astype(str))),'motif_axis_sha256':digest(list(read_elem(a['var']).index.astype(str)))}
  for rec in [*plan['inputs'].values(),plan['metadata'],plan['mapping']]:verify_record(root,rec)
  validate_plan(plan,fixture=_fixture)
  if bound_plan(root,plan_path,plan_sha)[1]!=plan_bytes:raise ValueError('Plan changed during export')
  if sha(ROOT/FOUNDATION)!=FOUNDATION_SHA or sha(ROOT/PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Reviewed source/proposal changed during export')
  report={'schema':'dynavelo_control_export_manifest_v1','status':'PASS_EXPORTED_INPUT_INTEGRITY_ONLY','plan_sha256':plan_sha,'control':plan['control'],'lineage':lineage,'axes':axes,'versions':versions,'velocity_nan_entries':nan_count,'official_mask_semantics':'~isnan(velocity), then nan_to_num in official Dataset','explicit_mask_layers':mask_names,'changed_h5_paths':changed,'unchanged_fields_verified_by':'complete logical HDF dataset/attribute SHA census excluding only listed changes and two lineage fields','outputs':{k:record(p,root) for k,p in outputs.items()},'full_mapping_audit':{k:v for k,v in mapping.items() if k!='pairs'},'training_authorized':False,'scientific_pass':False,'fixture_only':_fixture}
  write_new(out/'export_manifest.json',report);write_new(out/'COMPLETE.json',{'schema':'dynavelo_control_export_complete_v1','manifest':record(out/'export_manifest.json',root),'plan_sha256':plan_sha,'scientific_pass':False});return report
 except BaseException as exc:
  write_new(out/'FAILED.json',{'status':'INCOMPLETE_NO_TRAINING_AUTHORITY','error_type':type(exc).__name__,'detail':str(exc),'plan_sha256':plan_sha});raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 path=a.plan.absolute();plan=validate_plan(bound_plan(ROOT,path,a.plan_sha256)[0])
 if not a.execute:print(json.dumps({'status':'PLAN_ONLY_NO_INPUT_OR_MAPPING_READ','control':plan['control'],'execution_authorized':plan.get('execution_authorized') is True,'human_mapping_created':False,'training_started':False}));return
 result=export_inputs(ROOT,path,a.plan_sha256);print(json.dumps({'status':result['status'],'control':result['control'],'outputs':result['outputs'],'training_authorized':False}))
if __name__=='__main__':main()
