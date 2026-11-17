"""Verify archived bytes using only the Python standard library; no science replay."""
from pathlib import Path, PurePosixPath
import argparse,csv,hashlib,json,zipfile

def safe_member(name):
    p=PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name

def verify(root):
    groups={}
    with (root/'FILE_MANIFEST.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            assert Path(r['archive']).name==r['archive'],'Unsafe archive name'
            assert safe_member(r['member']),'Unsafe member name'
            group=groups.setdefault(r['archive'],{})
            assert r['member'] not in group,'Duplicate manifest member'
            group[r['member']]=r
    results=[]
    for name,entries in groups.items():
        path=root/name
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            assert len(names)==len(set(names)),'Duplicate ZIP member'
            assert all(safe_member(n) for n in names),'Unsafe ZIP member'
            extra=set(names)-set(entries)
            assert all(n.startswith('DynaVelo_Research/deposit_manifests/') for n in extra),extra
            assert set(entries)<=set(names),'Missing payload'
            for member,r in entries.items():
                info=z.getinfo(member);assert info.file_size==int(r['bytes']),(name,member,'size')
                h=hashlib.sha256()
                with z.open(member) as stream:
                    for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
                assert h.hexdigest()==r['sha256'],(name,member,'sha256')
            # Reading each payload to EOF above also verifies its ZIP CRC.
            for n in extra:z.read(n)
        result={'archive':name,'payload_files':len(entries),'sha256_and_crc':'pass'}
        results.append(result);print(json.dumps(result),flush=True)
    sums=root/'SHA256SUMS.tsv'
    if sums.exists():
        with sums.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                assert Path(r['file']).name==r['file']
                p=root/r['file']
                assert p.stat().st_size==int(r['bytes']),p.name
                with p.open('rb') as s:assert hashlib.file_digest(s,'sha256').hexdigest()==r['sha256'],p.name
    return {'archives':results,'total_payload_files':sum(len(v) for v in groups.values()),'analysis_executed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deposit',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=verify(args.deposit.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS: all deposited payload bytes verified; no scientific analysis executed.')
