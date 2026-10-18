"""M2-A read-only numeric, provenance, font and PDF geometry verification."""
import json
import re
import pdfplumber
from pypdf import PdfReader
from reportlab.pdfbase.ttfonts import TTFont
from src.export.academic_report import ROOT,OUT,PDF,MD,FONT
from src.export.c3 import load_json
from src.training.c1 import C1Gate
from src.training.b3_io import sha
from src.training.checkpoint import read_checkpoint


def normalized(text):return re.sub(r'\s+','',text).replace(',','')


def qa():
    with C1Gate('none') as gate:
        m=load_json(OUT/'report_manifest.json')
        index=load_json(OUT/'source_index.json')
        structure=load_json(OUT/'document_structure.json')
        facts=load_json(OUT/'fact_ledger.json')
        reader=PdfReader(PDF)
        text='\n'.join(page.extract_text() or '' for page in reader.pages)
        plain=normalized(text)
        checks={}
        checks['all_report_hashes']=all(sha(ROOT/path)==value for path,value in m['files'].items())
        checks['all_source_hashes']=all(sha(ROOT/path)==value for path,value in m['inputs'].items())
        checks['logical_and_pdf_pages_match']=len(reader.pages)==len(structure)==m['pages']
        checks['each_heading_on_own_page']=all(normalized(p['title']) in normalized(reader.pages[i].extract_text()) for i,p in enumerate(structure))
        checks['thirteen_figures_twenty_three_tables']=len(index['figures'])==13 and len(index['tables'])==23
        checks['every_figure_caption_extracted']=all(b['id'] in text for b in index['figures'])
        checks['figure_source_and_image_hashes']=all(sha(ROOT/b['source'])==b['source_sha256'] and sha(ROOT/b['path'])==m['inputs'][b['path']] for b in index['figures'])
        numeric_errors=[];display_errors=[]
        for f in facts:
            path=ROOT/f['source']
            value=read_checkpoint(path) if 'payload' in load_json(path) else load_json(path)
            for key in f['selector']:value=value[key]
            if value!=f['value']:numeric_errors.append(f['name'])
            if normalized(f['display']) not in plain:display_errors.append(f['name'])
        checks['fact_ledger_matches_sources']=not numeric_errors
        checks['all_fact_displays_in_pdf']=not display_errors
        checks['six_crispdm_stages']=all(v in text for v in ['Business Understanding','Data Understanding','Data Preparation','Modeling','Evaluation','Deployment'])
        checks['limitations_present']=all(v in text for v in ['未來','交換性','高耗電','尚未實作','一次','0.077777'])
        checks['no_replacement_character']='\ufffd' not in text
        refs=load_json(ROOT/'docs/research/references.json')
        checks['references_verified_metadata']=len(refs)==9 and all(r['verified']=='2026-10-17' and r['url'].startswith('https://') and '['+r['id']+']' in text for r in refs)
        font=TTFont('QA',str(FONT),subfontIndex=0)
        intended=''.join(p['title']+''.join(b.get('text','')+b.get('title','') for b in p['blocks']) for p in structure)
        unsupported=sorted(set(c for c in intended if not c.isspace() and ord(c) not in font.face.charToGlyph))
        checks['all_authored_glyphs_supported']=not unsupported
        embedded=[]
        for page in reader.pages:
            for f in page['/Resources']['/Font'].values():
                f=f.get_object();descriptor=f.get('/FontDescriptor')
                if descriptor and '/FontFile2' in descriptor.get_object():embedded.append(str(f['/BaseFont']))
        checks['embedded_regular_and_bold_truetype']=len(set(embedded))>=2
        bad_bounds=[];page_geometry=[]
        with pdfplumber.open(PDF) as document:
            for i,page in enumerate(document.pages):
                for c in page.chars:
                    if c['x0']<10 or c['x1']>page.width-10 or c['top']<10 or c['bottom']>page.height-10:
                        bad_bounds.append(dict(page=i+1,text=c['text'],x0=c['x0'],x1=c['x1'],top=c['top'],bottom=c['bottom']))
                page_geometry.append(dict(page=i+1,width=page.width,height=page.height,characters=len(page.chars),images=len(page.images)))
        checks['all_text_inside_safe_page_bounds']=not bad_bounds
        checks['no_blank_pages']=all(p['characters']>100 for p in page_geometry)
        old=read_checkpoint(ROOT/'reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json')
        checks['frozen_model_hashes_unchanged']=all(sha(ROOT/path)==value for path,value in old['hashes_after'].items())
        checks['test_official_predict_count_unchanged']=old['test_prediction_calls']==1
        result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,pages=len(reader.pages),
            figures=len(index['figures']),tables=len(index['tables']),numeric_facts=len(facts),numeric_errors=numeric_errors,
            display_errors=display_errors,unsupported_glyphs=unsupported,bad_bounds=bad_bounds,
            embedded_fonts=sorted(set(embedded)),page_geometry=page_geometry,source_dataset_reads=gate.reads,
            model_prediction_calls=0,visual_review='Recorded separately after inspecting rendered PNG pages')
        (OUT/'pdf_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        m['status']='QA_COMPLETE' if result['status']=='PASS' else 'QA_FAILED'
        m['qa']=dict(path='reports/academic/m2a-v1/pdf_qa.json',sha256=sha(OUT/'pdf_qa.json'),
                     checks=len(checks),numeric_facts=len(facts))
        (OUT/'report_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        return result


if __name__=='__main__':
    result=qa();print(json.dumps({k:v for k,v in result.items() if k!='page_geometry'},ensure_ascii=False,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
