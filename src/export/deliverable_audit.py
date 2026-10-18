"""M2-B read-only deliverable audit. Never fabricate evidence or create a ZIP."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from PIL import Image
from src.training.c1 import ROOT,C1Gate
from src.training.b3_io import sha,write_once,now
from src.training.checkpoint import read_checkpoint

OUT=ROOT/'reports/deliverables/m2b'
STATES={'COMPLETE_VERIFIED','COMPLETE_UNVERIFIED','USER_ACTION_REQUIRED','PENDING'}
FROZEN_EVIDENCE=[
 ('reports/experiments/b2b-matrix-v1/events.jsonl','RUNTIME_EVENT_LOG'),
 ('reports/experiments/b2b-matrix-v1/manifest.json','EXPERIMENT_MANIFEST'),
 ('reports/experiments/b3-validation-v1/candidate_lock.json','PRE_VALIDATION_LOCK'),
 ('reports/experiments/b3-validation-v1/mlr_decision.json','MODEL_SELECTION_DECISION'),
 ('reports/experiments/c1-mlr-1.0.0/final_fit.json','FROZEN_FINAL_FIT_RECORD'),
 ('reports/experiments/c1-mlr-1.0.0/calibration.json','CALIBRATION_RECORD'),
 ('reports/experiments/c1-mlr-1.0.0/manifest.json','MODEL_MANIFEST'),
 ('reports/evaluation/c2-mlr-1.0.0/first_evaluation.json','ONE_SHOT_TEST_LEDGER'),
 ('reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json','TEST_EVALUATION_MANIFEST'),
 ('reports/evaluation/c3-mlr-1.0.0/consolidation_manifest.json','CONSOLIDATION_MANIFEST'),
 ('reports/academic/m2a-v1/report_manifest.json','REPORT_MANIFEST'),
 ('docs/technical/IMPLEMENTATION_PROGRESS.md','RETROSPECTIVE_PROGRESS_SUMMARY_NOT_TRANSCRIPT')]


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_relative(base,name):
    if not isinstance(name,str) or not name or '\\' in name:raise ValueError('Nonempty POSIX relative path required')
    part=PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or re.match(r'^[A-Za-z]:',name):raise ValueError('Unsafe relative path')
    path=(Path(base)/name).resolve()
    if not path.is_relative_to(Path(base).resolve()):raise ValueError('Path escapes evidence directory')
    return path


def inspect_pdf(path):
    path=Path(path)
    if path.stat().st_size>100_000_000:raise ValueError('PDF requires manual large-file review')
    with path.open('rb') as stream:
        if stream.read(5)!=b'%PDF-':raise ValueError('Not a PDF file')
    reader=PdfReader(path)
    if reader.is_encrypted:raise ValueError('Encrypted PDF cannot be audited')
    if not len(reader.pages):raise ValueError('PDF has no pages')
    text='\n'.join(p.extract_text() or '' for p in reader.pages)
    return dict(bytes=path.stat().st_size,pages=len(reader.pages),sha256=sha(path),
        extractable_characters=len(text),replacement_characters=text.count('\ufffd')),text


def file_ref(base,record):
    path=safe_relative(base,record['path'])
    if not path.is_file() or path.stat().st_size==0:raise ValueError('Missing/empty user-provided evidence')
    if sha(path)!=str(record['sha256']).lower():raise ValueError('Received evidence SHA-256 mismatch')
    return path


def stamp(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('Evidence date needs timezone')
    return dt


def reviewed(record,fields):
    if not record.get('reviewed_by'):raise ValueError('Human reviewer missing')
    stamp(record['reviewed_at'])
    if not all(record.get(field) is True for field in fields):raise ValueError('Human authenticity/completeness review not confirmed')


def canonical_text(text):return re.sub(r'\s+','',text)


def is_deliverable_file(path):
    path=Path(path)
    return path.is_file() and path.stat().st_size>0


def notebooklm_status(base):
    base=Path(base);metadata=base/'evidence.json'
    missing=dict(status='USER_ACTION_REQUIRED',reason='No verified genuine NotebookLM export received',received=False)
    if not metadata.exists():return missing
    try:
        m=load(metadata)
        if m.get('is_template') is not False or m.get('origin')!='USER_PROVIDED_NOTEBOOKLM_EXPORT':raise ValueError('Template is not NotebookLM evidence')
        if not m.get('notebook_name'):raise ValueError('Notebook identity missing')
        stamp(m['used_at'])
        reviewed(m['human_review'],['authentic_notebooklm_use','traditional_chinese_confirmed','citations_checked','summary_matches_original_output','privacy_reviewed'])
        original=file_ref(base,m['original_export']);summary=file_ref(base,m['summary_text']);pdf=file_ref(base,m['summary_pdf'])
        body=summary.read_text(encoding='utf-8-sig').strip()
        if 'TEMPLATE_ONLY' in body:raise ValueError('Template text cannot count as summary')
        count=len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]',body))
        if count<100:raise ValueError('Summary body needs at least100 Han characters')
        info,pdftext=inspect_pdf(pdf)
        if original.suffix.lower()=='.pdf':_,raw=inspect_pdf(original)
        elif original.suffix.lower() in ['.txt','.md']:raw=original.read_text(encoding='utf-8-sig')
        else:raise ValueError('Provide a text/Markdown/PDF original export for audit')
        if canonical_text(body) not in canonical_text(raw) or canonical_text(body) not in canonical_text(pdftext):
            raise ValueError('Summary text must be retained verbatim in original export and summary PDF')
        ids={r['id'] for r in m['sources']}
        if not ids or len(ids)!=len(m['sources']):raise ValueError('Unique genuine source list required')
        for source in m['sources']:
            if not source.get('title') or not (source.get('url') or source.get('local_file')):raise ValueError('Source location missing')
            if source.get('local_file'):file_ref(base,source['local_file'])
        if not m['source_alignment']:raise ValueError('Source alignment required')
        for row in m['source_alignment']:
            if row['source_id'] not in ids or not row.get('summary_section') or not row.get('locator'):raise ValueError('Invalid source alignment')
        if not m['screenshots']:raise ValueError('Genuine use screenshot required')
        for image in m['screenshots']:
            with Image.open(file_ref(base,image)) as opened:opened.verify()
        return dict(status='COMPLETE_VERIFIED',received=True,han_characters=count,pdf=info,
            authenticity='USER_ATTESTED_WITH_ORIGINAL_EXPORT_AND_SOURCE_ALIGNMENT; not independent platform authentication',
            traditional_chinese='HUMAN_CONFIRMED; Han count alone does not distinguish simplified/traditional')
    except Exception as error:
        return dict(status='USER_ACTION_REQUIRED',reason=str(error),received=True,verification='INCOMPLETE_OR_INVALID_EVIDENCE')


def conversation_status(base):
    base=Path(base);path=base/'index.json'
    if not path.exists():return dict(status='USER_ACTION_REQUIRED',received=False,reason='Three categories of genuine original exports not received')
    try:
        m=load(path)
        if m.get('is_template') is not False or m.get('origin')!='USER_PROVIDED_ORIGINAL_EXPORTS':raise ValueError('Template is not a conversation export')
        categories=set();results=[]
        for thread in m['threads']:
            if not thread.get('title') or not thread.get('platform') or not thread.get('export_method'):raise ValueError('Original conversation provenance missing')
            stamp(thread['exported_at'])
            reviewed(thread['human_review'],['authentic_export','messages_unedited','chronology_preserved','context_included','privacy_reviewed'])
            original=file_ref(base,thread['original']);pdf=file_ref(base,thread['pdf'])
            info,_=inspect_pdf(pdf)
            categories.add(thread['category']);results.append(dict(category=thread['category'],pdf=info,original_sha256=sha(original)))
        if not {'planning_pdd','supervision_prompts','coding_agent'}<=categories:raise ValueError('All three conversation categories required')
        return dict(status='COMPLETE_VERIFIED',received=True,threads=results,
            authenticity='USER_ATTESTED_ORIGINAL_EXPORTS; file validation cannot prove message completeness or platform origin')
    except Exception as error:
        return dict(status='USER_ACTION_REQUIRED',received=True,reason=str(error),verification='INCOMPLETE_OR_INVALID_EVIDENCE')


def junit_info(path):
    root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else root.findall('testsuite')
    tests=sum(int(s.get('tests',0)) for s in suites);fail=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    skipped=sum(int(s.get('skipped',0)) for s in suites)
    return dict(tests=tests,failures_errors=fail,skipped=skipped,status='FAIL' if fail else 'PASS' if tests and not skipped else 'NOT_RUN',
                contains_machine_hostname=any(bool(s.get('hostname')) for s in suites))


def evidence_index(root,output):
    entries=[]
    for name,kind in FROZEN_EVIDENCE:
        path=root/name
        if path.exists():entries.append(dict(path=name,kind=kind,bytes=path.stat().st_size,sha256=sha(path)))
    for path in sorted((root/'reports/tests').glob('*.xml')):
        entries.append(dict(path=str(path.relative_to(root)).replace('\\','/'),kind='ORIGINAL_JUNIT_EXECUTION_RESULT',
                            bytes=path.stat().st_size,sha256=sha(path),**junit_info(path)))
    result=dict(scope='CODING_OPERATION_EVIDENCE_INDEX_NOT_CONVERSATION',at_utc=now(),entries=entries,
        original_transcript_included=False,limitations='Runtime events, checkpoints, decisions and JUnit are real execution evidence. Progress summaries and this index are retrospective organization, not original user/AI messages. Historical failed tests remain labelled FAIL.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Coding Agent開發歷程證據索引','',
        '**這是實際檔案的後設索引，不是原始使用者／AI對話，也不替代完整Coding Agent對話PDF。**','',
        '| 原始位置 | 類型 | 實際測試狀態 | SHA-256 |','|---|---|---|---|']
    lines += [f"| {r['path']} | {r['kind']} | {r.get('status','不適用')} | {r['sha256']} |" for r in entries]
    lines += ['',result['limitations']]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def forbidden_package_path(source,archive):
    return any(part.startswith('.venv') or part in ['__pycache__','.pytest_cache','.git','incoming']
        for part in PurePosixPath(source).parts) or source.startswith(('data/raw/','data/processed/','tmp/')) or source.endswith('.joblib')


def security_scan(root,files):
    findings=[]
    token=re.compile(r'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    absolute=re.compile(r'(?i)(?:C:[\\/]+Users[\\/]+(?!someone|user)[^\s"<>]+|/home/(?!user/)[^\s"<>]+)')
    for name in sorted(set(files)):
        path=root/name
        if path.suffix.lower() not in ['.py','.md','.txt','.json','.xml','.csv','.pdf']:continue
        if path.stat().st_size>10_000_000:
            findings.append(dict(path=name,kind='LARGE_FILE_REVIEW',severity='MANUAL_REQUIRED'));continue
        text=inspect_pdf(path)[1] if path.suffix.lower()=='.pdf' else path.read_text(encoding='utf-8-sig',errors='replace')
        if token.search(text):findings.append(dict(path=name,kind='CREDENTIAL_SHAPED_VALUE',severity='FAIL',value_not_logged=True))
        if absolute.search(text):findings.append(dict(path=name,kind='PERSONAL_PATH_REVIEW',severity='MANUAL_REQUIRED',value_not_logged=True))
        if path.suffix=='.xml' and junit_info(path)['contains_machine_hostname']:
            findings.append(dict(path=name,kind='JUNIT_MACHINE_METADATA',severity='MANUAL_REQUIRED',value_not_logged=True))
    return dict(scope='PLANNED_ASSIGNMENT_TEXT_FILES_ONLY; not a universal PII detector',findings=findings,
        high_confidence_secret_matches=sum(r['severity']=='FAIL' for r in findings),
        policy='Original evidence remains unchanged. Incoming files stay private/gitignored. Website uses C3 public allowlist only. Human review of PDF/image personal information is still required.')


def run(root=ROOT,output=OUT):
    root,output=Path(root),Path(output)
    with C1Gate('none') as gate:
        plan=load(root/'configs/deliverables.json')
        coding=evidence_index(root,output/'coding_evidence_index.json')
        items=[];mappings=[];errors=[]
        for spec in plan['items']:
            item=dict(spec)
            if spec['kind']=='manual':item['status']=spec['status'];item['verification']='NOT_AUTOMATICALLY_CONFIRMED'
            elif spec['kind']=='notebooklm':item.update(notebooklm_status(root/'deliverables/incoming/notebooklm'))
            elif spec['kind']=='conversations':item.update(conversation_status(root/'deliverables/incoming/conversations'))
            else:
                paths=sorted(root.glob(spec['source'])) if spec['kind']=='glob' else [safe_relative(root,spec['source'])]
                paths=[p for p in paths if is_deliverable_file(p)]
                item['files']=[dict(path=str(p.relative_to(root)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
                item['status']=('COMPLETE_UNVERIFIED' if spec['kind']=='unverified_file' else 'COMPLETE_VERIFIED') if paths else spec.get('missing_status','PENDING')
                if spec['kind']!='report_pdf':
                    for path in paths:
                        if path.suffix.lower()=='.pdf':inspect_pdf(path)
                if spec['kind']=='report_pdf' and paths:
                    info,_=inspect_pdf(paths[0]);item['pdf']=info
                    report=load(root/'reports/academic/m2a-v1/report_manifest.json')
                    expected=report['files'][spec['source']]
                    if info['sha256']!=expected or info['pages']!=report['pages']:raise ValueError('Formal report hash/page mismatch')
                    qa=load(root/'reports/academic/m2a-v1/pdf_qa.json')
                    visual=read_checkpoint(root/'reports/academic/m2a-v1/visual_review.json')
                    if qa['status']!='PASS' or visual['pdf_sha256']!=info['sha256']:raise ValueError('Formal PDF QA evidence mismatch')
                    item['font_layout_qa']='M2A_AUTOMATED_PASS_AND_VISUAL_REVIEW_HASH_MATCH'
                for p in paths:
                    source=str(p.relative_to(root)).replace('\\','/')
                    target=spec['archive']+(source[len(spec['strip_prefix']):] if spec['kind']=='glob' else '')
                    safe_relative(root,target)
                    if forbidden_package_path(source,target):errors.append('Forbidden package member: '+source)
                    mappings.append(dict(item=spec['id'],source=source,archive=target,bytes=p.stat().st_size,sha256=sha(p)))
            if item['status'] not in STATES:raise ValueError('Unknown deliverable state')
            items.append(item)
        destinations=[r['archive'].casefold() for r in mappings]
        if len(destinations)!=len(set(destinations)):errors.append('Duplicate case-insensitive archive paths')
        if any(r['source'].startswith('deliverables/templates/') for r in mappings):errors.append('Template must not count as final evidence')
        frozen=read_checkpoint(root/'reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json')
        for name,expected in frozen['hashes_after'].items():
            if sha(root/name)!=expected:errors.append('Frozen artifact changed: '+name)
        safety=security_scan(root,[r['source'] for r in mappings])
        if safety['high_confidence_secret_matches']:errors.append('Review credential-shaped matches before packaging')
        source_count=len(list((root/'src').rglob('*.py')))
        import os
        notebooks=[]
        for folder,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if not d.startswith('.venv') and d not in ['.git','__pycache__','.pytest_cache']]
            notebooks.extend(Path(folder)/name for name in files if name.endswith('.ipynb'))
        result=dict(audit_status='FAIL' if errors else 'PASS_WITH_OPEN_ITEMS',at_utc=now(),
            assignment_ready=False,final_zip_created=False,release_gate='NOT_RUN',website='PENDING',deployment='PENDING',
            authority=plan['authority'],course_requirements='USER_ACTION_REQUIRED_ORIGINAL_NOT_RECEIVED',
            items=items,status_counts=dict(Counter(r['status'] for r in items)),planned_members=mappings,
            planned_archive_bytes=sum(r['bytes'] for r in mappings),security=safety,errors=errors,
            python_inventory=dict(py_files=source_count,ipynb_files=len(notebooks),ipynb_policy='Not fabricated; teacher requirement unconfirmed'),
            evidence_index=dict(path=str((output/'coding_evidence_index.json').relative_to(root)).replace('\\','/'),entries=len(coding['entries'])),
            source_dataset_reads=gate.reads,model_prediction_calls=0,
            limitations=['Integrity checks cannot independently prove NotebookLM/platform origin or complete conversation content.',
                'Planned relative paths are checked, but the ZIP has not been assembled/extracted or executed.',
                'Historical progress snapshots may have earlier hashes; current index pins the current summary, not a full transcript.'])
        output.mkdir(parents=True,exist_ok=True)
        runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write_once(output/f'audit-{runid}.json',result)
        (output/'latest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        lines=['# 作業交付檢查表（M2-B）','',f"稽核：{result['audit_status']}；ASSIGNMENT_READY=false；未製作最終ZIP。",'',
            '| 項目 | 狀態 | 原始位置 | 預定Archive位置 |','|---|---|---|---|']
        lines += [f"| {r['title']} | {r['status']} | {r.get('source','')} | {r.get('archive','')} |" for r in items]
        lines += ['', 'COMPLETE_VERIFIED僅代表所列檔案／完整性檢查通過，不把模板或摘要當原始平台紀錄。',
            'COMPLETE_UNVERIFIED表示已有草稿但未做解壓／執行驗證。USER_ACTION_REQUIRED為人工真實來源／核對缺件，PENDING為尚未完成工作。',
            '', '網站、GitHub Pages、JS／Browser驗收、最終ZIP均PENDING。詳細人工操作：deliverables/USER_ACTION_REQUIRED.md。']
        (root/'deliverables/ASSIGNMENT_CHECKLIST.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        structure=['# 預定ZIP目錄與精確檔案對照','', '僅計畫，沒有建立ZIP；缺件不以空檔／模板取代。','',
            '```text',plan['zip_root']+'/',*['  '+r['archive']+' <- '+r['source'] for r in mappings],'```','',
            '尚缺／人工／未驗證項目：']
        structure += ['- '+r['id']+'：'+r['status']+'；預定 '+r.get('archive','') for r in items if r['status']!='COMPLETE_VERIFIED']
        (root/'deliverables/ZIP_STRUCTURE_AND_FILE_MAP.md').write_text('\n'.join(structure)+'\n',encoding='utf-8')
        return result


if __name__=='__main__':
    result=run();print(json.dumps({k:result[k] for k in ['audit_status','status_counts','assignment_ready','final_zip_created','python_inventory','errors']},ensure_ascii=False,indent=2))
    if result['audit_status']=='FAIL':raise SystemExit(1)
