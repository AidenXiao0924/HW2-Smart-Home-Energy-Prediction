"""Render existing PDD and a labelled derivative decision index; never conversations."""
from html import escape
import json
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from pypdf import PdfReader
import pdfplumber
from src.training.c1 import ROOT,C1Gate
from src.training.b3_io import sha,write_once,now
from src.training.checkpoint import read_checkpoint
from src.export.deliverable_audit import evidence_index,OUT,junit_info

PDD=ROOT/'deliverables/prepared/pdd/Final_PDD_v1.0.pdf'
DECISION=ROOT/'deliverables/prepared/ai_collaboration/AI_Decision_Log.pdf'


def math_fallback(text):
    return text.replace('10⁻⁶','10^(-6)').replace('−','-').replace('≤','<=')


def render_markdown(text,path,title):
    text=math_fallback(text)
    pdfmetrics.registerFont(TTFont('EvidenceTC','C:/Windows/Fonts/msjh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('EvidenceBold','C:/Windows/Fonts/msjhbd.ttc',subfontIndex=0))
    cmap=pdfmetrics.getFont('EvidenceTC').face.charToGlyph
    unsupported=sorted(set(hex(ord(c)) for c in text if not c.isspace() and ord(c) not in cmap))
    if unsupported:raise ValueError('Unsupported PDF glyphs: '+','.join(unsupported))
    styles={
        'body':ParagraphStyle('Body',fontName='EvidenceTC',fontSize=9.6,leading=15,spaceAfter=6,wordWrap='CJK'),
        'cell':ParagraphStyle('Cell',fontName='EvidenceTC',fontSize=7.8,leading=11,wordWrap='CJK'),
        'code':ParagraphStyle('Code',fontName='EvidenceTC',fontSize=7.8,leading=11,spaceAfter=1,wordWrap='CJK'),
        'h1':ParagraphStyle('H1',fontName='EvidenceBold',fontSize=17,leading=22,spaceBefore=13,spaceAfter=12,wordWrap='CJK',keepWithNext=True),
        'h2':ParagraphStyle('H2',fontName='EvidenceBold',fontSize=13,leading=18,spaceBefore=12,spaceAfter=8,wordWrap='CJK',keepWithNext=True),
        'h3':ParagraphStyle('H3',fontName='EvidenceBold',fontSize=11,leading=16,spaceBefore=9,spaceAfter=6,wordWrap='CJK',keepWithNext=True)}
    def clean(value):return re.sub(r'[`*]','',value)
    def p(value,style='body'):return Paragraph(escape(clean(value)),styles[style])
    lines=text.splitlines();story=[];i=0;code=False
    while i<len(lines):
        line=lines[i];i+=1
        if line.strip()=='<!-- PAGE -->':story.append(PageBreak());continue
        if line.startswith('```'):code=not code;continue
        if code:
            story.append(p(line,'code'));continue
        if not line.strip():story.append(Spacer(1,3));continue
        if line.strip()=='---':story.append(Spacer(1,7));continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            rows=[r for r in rows if not re.fullmatch(r'[|\s:\-]+',r)]
            cells=[[p(c.strip(),'cell') for c in r.strip().strip('|').split('|')] for r in rows]
            cols=len(cells[0]);cells=[r+[p('','cell')]*(cols-len(r)) for r in cells]
            table=Table(cells,colWidths=[474/cols]*cols,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9ef')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f4f7f8'),colors.white]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([table,Spacer(1,8)]);continue
        heading=re.match(r'^(#{1,6})\s+(.*)',line)
        story.append(p(heading.group(2),'h'+str(min(3,len(heading.group(1))))) if heading else p(line))
    def chrome(canvas,doc):
        canvas.saveState();canvas.setFont('EvidenceTC',7.5);canvas.setFillColor(colors.HexColor('#536774'))
        canvas.drawString(20*mm,A4[1]-12*mm,title)
        canvas.drawString(20*mm,10*mm,'M2-B / 來源與狀態以原始檔案及Manifest為準')
        canvas.drawRightString(A4[0]-20*mm,10*mm,str(doc.page));canvas.restoreState()
    path.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=21*mm,bottomMargin=20*mm,
        title=title,author='Smart Home Energy Research Project').build(story,onFirstPage=chrome,onLaterPages=chrome)


def inspect(path):
    reader=PdfReader(path);fonts=set();outside=[]
    for page in reader.pages:
        for font in page['/Resources']['/Font'].values():
            font=font.get_object();desc=font.get('/FontDescriptor')
            if desc and '/FontFile2' in desc.get_object():fonts.add(str(font['/BaseFont']))
    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages):
            if any(c['x0']<10 or c['x1']>page.width-10 or c['top']<10 or c['bottom']>page.height-10 for c in page.chars):outside.append(i+1)
    text='\n'.join(page.extract_text() for page in reader.pages)
    return dict(pages=len(reader.pages),bytes=path.stat().st_size,sha256=sha(path),embedded_fonts=sorted(fonts),
                outside_text_pages=outside,replacement_characters=text.count('\ufffd')),text


def build():
    destination=OUT/'supporting_pdf_manifest.json'
    if destination.exists():
        result=read_checkpoint(destination)
        for path,value in result['outputs'].items():
            if sha(ROOT/path)!=value:raise ValueError('Supporting PDF changed')
        return result
    with C1Gate('none') as gate:
        index=evidence_index(ROOT,OUT/'coding_evidence_index.json')
        decision=read_checkpoint(ROOT/'reports/experiments/b3-validation-v1/mlr_decision.json')
        final=read_checkpoint(ROOT/'reports/experiments/c1-mlr-1.0.0/final_fit.json')
        cal=read_checkpoint(ROOT/'reports/experiments/c1-mlr-1.0.0/calibration.json')
        test=read_checkpoint(ROOT/'reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json')
        raw=ROOT/'docs/pdd/Final_PDD_v1.0.md'
        pdd='\n'.join(['# Final PDD v1.0 - 原稿排版轉換版','',
            '以下內容由正式Markdown原稿直接排版；未更新其原始Implementation Status或勾選框。原稿狀態字樣不代表目前作業已完成。',
            '排版註記：為字型支援將少數數學符號等價顯示為ASCII（減號、<=、10^(-6)）；原始Markdown保持不變，未改規格意義。',
            '來源：docs/pdd/Final_PDD_v1.0.md；SHA-256：'+sha(raw),'','<!-- PAGE -->','',raw.read_text(encoding='utf-8')])
        render_markdown(pdd,PDD,'Final PDD v1.0 / 原稿排版版')
        lines=['# AI Decision Log與Coding Agent操作證據索引','',
            '**衍生整理文件，不是完整原始對話PDF。** 本文件於M2-B讀取既有真實Decision／Manifest／JUnit後排版。沒有重建使用者或AI訊息，沒有NotebookLM輸出。',
            '原始規劃、監督與Coding Agent對話仍USER_ACTION_REQUIRED；此PDF不能替代它們。',
            '', '## 正式模型決策','',
            '| 欄位 | 原始紀錄值 |','|---|---|',f"| Model Version | {decision['planned_model_version']} |",
            f"| Estimator | {decision['model_type']} |",f"| Method / K | {decision['method']} / {decision['k']} |",
            f"| Decision UTC | {decision['decided_at_utc']} |",f"| Validation RMSE (Wh) | {decision['validation_metrics']['rmse']} |",'',
            '凍結特徵：'+', '.join(decision['features']),
            '原始決策理由（逐字保留欄位內容）：'+decision['rationale'],
            '來源：reports/experiments/b3-validation-v1/mlr_decision.json','', '<!-- PAGE -->','',
            '# Final Fit、Calibration與一次性Test證據','',
            '| 紀錄 | 原始值 |','|---|---|',f"| Final Fit rows | {final['training']['rows']} |",
            f"| Final Fit period | {final['training']['start']} to {final['training']['end']} |",
            f"| Calibration n / k / alpha | {cal['n']} / {cal['k']} / {cal['alpha']} |",f"| Conformal q (Wh) | {cal['q']} |",
            f"| Test Run ID | {test['run_id']} |",f"| Test prediction calls | {test['test_prediction_calls']} |",
            f"| Test RMSE (Wh) | {test['metrics']['rmse']} |",f"| Test coverage_all | {test['prediction_interval']['coverage_all']} |",'',
            '來源：C1 final_fit.json、calibration.json；C2 evaluation_manifest.json。以上是保存結果，不是M2-B重新評估。',
            '觀察到的Test Coverage不是未來時間區段嚴格95%保證。模型、特徵與q未修改。','',
            '## 原始操作紀錄與後來摘要的區別','',
            'RUNTIME_EVENT_LOG、Decision、Checkpoint Manifest與JUnit為保存的實際執行證據。IMPLEMENTATION_PROGRESS為後來整理的進度摘要。本文件與coding_evidence_index也是衍生索引；都不是完整使用者／AI原始訊息。',
            '', '<!-- PAGE -->','', '# 原始證據檔案索引','', '| 來源 | 類型／狀態 | SHA-256 |','|---|---|---|']
        for r in index['entries']:
            lines.append(f"| {r['path']} | {r['kind']} / {r.get('status','N/A')} | {r['sha256']} |")
        lines += ['', '歷史FAIL測試保留原始標記，不以後來PASS掩蓋。JUnit可能含機器資訊；原始紀錄不改寫，提交前仍需人工隱私確認。',
            '完整檔案索引與目前狀態請見reports/deliverables/m2b/coding_evidence_index.json；此PDF是產生時快照。']
        decision_md=OUT/'AI_Decision_Log.md';decision_md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        render_markdown('\n'.join(lines),DECISION,'AI Decision Log / 衍生整理，非原始對話')
        qas={}
        for path in [PDD,DECISION]:
            info,text=inspect(path)
            if info['outside_text_pages'] or info['replacement_characters'] or not info['embedded_fonts']:raise ValueError('Supporting PDF QA failed')
            qas[str(path.relative_to(ROOT)).replace('\\','/')]=info
        result=dict(status='AUTOMATED_QA_PASS_PENDING_VISUAL_REVIEW',at_utc=now(),
            pdd_source_sha256=sha(raw),decision_is_original_conversation=False,qa=qas,source_dataset_reads=gate.reads,
            inputs={r['path']:r['sha256'] for r in index['entries']},
            outputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [PDD,DECISION,decision_md]},
            code_sha256=sha(Path(__file__)))
        write_once(destination,result)
        return result


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
