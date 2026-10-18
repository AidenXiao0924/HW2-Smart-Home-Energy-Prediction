"""M2-A report authoring from frozen results only; no data/model inference."""
import argparse
from copy import deepcopy
from html import escape
import json
from importlib.metadata import version
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from pypdf import PdfReader
from src.export.c3 import ROOT,PUBLIC,PRIVATE,load_json,csv
from src.export.c3_validate import verify
from src.training.c1 import C1Gate,ARTIFACT
from src.training.b3_io import sha,now
from src.training.checkpoint import read_checkpoint

OUT=ROOT/'reports/academic/m2a-v1'
PDF=ROOT/'output/pdf/HW2_CRISP_DM_Research_Report.pdf'
MD=ROOT/'docs/research/HW2_CRISP_DM_Research_Report.md'
FONT=Path('C:/Windows/Fonts/msjh.ttc')
BOLD=Path('C:/Windows/Fonts/msjhbd.ttc')


def fmt(value,d=4):return f'{float(value):.{d}f}'


class Author:
    def __init__(self):self.pages=[];self.figures=[];self.tables=[];self.facts=[];self.inputs={}
    def source(self,path):
        path=Path(path);name=str(path.relative_to(ROOT)).replace('\\','/')
        self.inputs[name]=sha(path);return name
    def page(self,title):
        self.pages.append(dict(title=title,blocks=[]));return self
    def p(self,text):self.pages[-1]['blocks'].append(dict(kind='paragraph',text=text));return self
    def table(self,title,headers,rows,source,split,widths=None,note=''):
        item=dict(kind='table',id=f'T{len(self.tables)+1:02d}',title=title,headers=headers,rows=rows,
            source=self.source(source),source_sha256=sha(source),split=split,widths=widths,note=note)
        self.tables.append(item);self.pages[-1]['blocks'].append(item);return self
    def figure(self,title,path,source,split,height=190,note=''):
        item=dict(kind='figure',id=f'F{len(self.figures)+1:02d}',title=title,path=self.source(path),
            source=self.source(source),source_sha256=sha(source),split=split,height=height,note=note)
        self.figures.append(item);self.pages[-1]['blocks'].append(item);return self
    def fact(self,name,value,source,selector,split,display=None):
        self.facts.append(dict(name=name,value=value,source=self.source(source),selector=selector,split=split,
                               display=str(value) if display is None else display))


def content():
    verify()
    a=Author()
    s=load_json(PUBLIC/'research_summary.json');model=load_json(ARTIFACT)
    refs=load_json(ROOT/'docs/research/references.json');a.source(ROOT/'docs/research/references.json')
    a.source(ROOT/'docs/pdd/Final_PDD_v1.0.md');a.source(ARTIFACT)
    quality_path=ROOT/'reports/data/quality_report.json';quality=load_json(quality_path)
    split_path=ROOT/'reports/data/split_manifest.json';split=load_json(split_path)
    cv_path=ROOT/'reports/evaluation/b2b/feature_selection_results.csv';cv=csv(cv_path)
    baseline_path=ROOT/'reports/evaluation/b2b/baselines_summary.csv';baselines=csv(baseline_path)
    val_path=ROOT/'reports/evaluation/b3/validation_metrics.csv';val=csv(val_path)
    test_path=ROOT/'reports/evaluation/c2-mlr-1.0.0/test_metrics.json';test=read_checkpoint(test_path)
    diag_path=ROOT/'reports/evaluation/c2-mlr-1.0.0/diagnostics.json';diag=read_checkpoint(diag_path)
    cal_path=ROOT/'reports/experiments/c1-mlr-1.0.0/calibration.json';cal=read_checkpoint(cal_path)
    final_path=ROOT/'reports/experiments/c1-mlr-1.0.0/final_fit.json';final=read_checkpoint(final_path)
    vif_path=PRIVATE/'train_validation_vif.json';vif=read_checkpoint(vif_path)
    lock_path=ROOT/'reports/experiments/b3-validation-v1/candidate_lock.json';lock=read_checkpoint(lock_path)
    decision_path=ROOT/'reports/experiments/b3-validation-v1/mlr_decision.json';decision=read_checkpoint(decision_path)
    c2=ROOT/'reports/evaluation/c2-mlr-1.0.0';b2fig=ROOT/'reports/figures/b2b'
    a.source(ROOT/'reports/data/provenance.json');a.source(ROOT/'configs/b3_protocol.json')
    metrics=test['metrics'];pi=test['prediction_interval'];bp=diag['breusch_pagan']
    for key in ['mae','mse','rmse','r2','adjusted_r2']:
        a.fact('Test '+key,metrics[key],test_path,['metrics',key],'TEST',fmt(metrics[key],6))
    for key in ['n','k','q','alpha']:a.fact('Calibration '+key,cal[key],cal_path,[key],'CALIBRATION')
    for key in ['total_rows','valid_intervals','invalid_intervals','covered_count','coverage_all','average_width_valid',
                'median_width_valid','negative_prediction_count','negative_warning_count','below_lower_count_valid','above_upper_count_valid']:
        display=(f"{100*pi[key]:.4f}%" if key=='coverage_all' else fmt(pi[key],6) if key in
                 ['average_width_valid','median_width_valid'] else str(pi[key]))
        a.fact('Test PI '+key,pi[key],test_path,['prediction_interval',key],'TEST',display)
    a.fact('DW',diag['durbin_watson'],diag_path,['durbin_watson'],'TEST',fmt(diag['durbin_watson'],6))
    a.fact('BP LM',bp['lm_statistic'],diag_path,['breusch_pagan','lm_statistic'],'TEST',fmt(bp['lm_statistic'],6))
    a.fact('BP LM p',bp['lm_pvalue'],diag_path,['breusch_pagan','lm_pvalue'],'TEST',f"{bp['lm_pvalue']:.6e}")
    for key in ['rows','columns']:a.fact('Raw '+key,quality[key],quality_path,[key],'ALL_RAW')
    for name,m in split['partitions'].items():a.fact(name+' rows',m['rows'],split_path,['partitions',name,'rows'],name.upper())
    for i,f in enumerate(model['feature_order']):a.fact('Coefficient '+f,model['coefficients'][i],ARTIFACT,['coefficients',i],'TRAIN_VALIDATION',fmt(model['coefficients'][i],8))

    a.page('智慧家庭能源消耗估計與分析').p('Smart Home Energy Consumption Prediction & Analytics').p(
        'CRISP-DM 學術研究報告 | 物聯網應用與資料分析 HW2 | 2026-10-17').p(
        '模型版本：mlr-1.0.0；研究階段：Milestone 2 - Part A。依 Final PDD v1.0 與已凍結研究產物撰寫。').p('摘要').p(
        f"本研究使用 Appliances Energy Prediction 真實資料，以同一觀測時間的感測器與氣象資訊估計家電能源消耗，而非預測未來時段。"
        f"資料含 {quality['rows']:,} 筆與 {quality['columns']} 個原始欄位，依時間切為 Train、Validation、Calibration 與 Test。"
        '五種 Feature Selection、兩種候選集合與五個 K 配置形成 50 組方案及 250 個外折結果。依預鎖定 Validation 比較，'
        '正式模型選用 Forward SFS 的 15 特徵 Multiple Linear Regression，於 Train+Validation 重新 Fit 後凍結。'
        f"Test RMSE 為 {fmt(metrics['rmse'],6)} Wh，R² 為 {fmt(metrics['r2'],6)}；結果僅顯示有限的解釋能力，不能稱為高準確度模型。"
        f"獨立 Calibration 取得 q={cal['q']} Wh，實際 Test Coverage 為 {100*pi['coverage_all']:.4f}%。"
        '殘差診斷呈現時間相依、右尾及高耗電峰值低估。時間序列不保證交換性，名目95%區間不代表未來嚴格覆蓋保證。'
        '模型與部署 JSON 契約已完成，網站功能尚未實作。').p('關鍵詞：CRISP-DM、Multiple Linear Regression、Feature Selection、TimeSeriesSplit、Split Conformal、智慧家庭。')

    a.page('閱讀導覽與研究證據').p('本報告以 CRISP-DM 六階段組織研究問題、資料處理、模型選擇、評估與部署規劃 [R2]。各章節採保存的真實產物，所有圖表均附資料區段與來源；完整 SHA-256 見來源索引及 report_manifest.json。').table(
        '六階段章節導覽',['階段','核心問題','本研究交付'],[
        ['1 Business Understanding','要估計什麼？','同時段耗電估計與可稽核成功標準'],
        ['2 Data Understanding','資料代表什麼？','來源、授權、欄位與品質'],['3 Data Preparation','如何隔離資訊？','時間切分與每折前處理'],
        ['4 Modeling','如何比較與選模？','五選擇器、50配置、Validation鎖定'],['5 Evaluation','結果可信到何種程度？','一次Test、區間及統計限制'],
        ['6 Deployment','如何交付並限制用途？','JSON推論契約；網站尚未實作']],ROOT/'docs/pdd/Final_PDD_v1.0.md','研究流程',widths=[135,105,240]).p(
        '數字閱讀規則：CV mean±SD 是五個外折的摘要；Validation 用於有限候選比較；Calibration 只估計固定 q；Test 是凍結後一次性評估。跨區段誤差不可視為同一母體的直接改善百分比。圖表樣本與小時彙整僅供展示，不用於重新計算正式指標。')

    a.page('1 Business Understanding：問題與成功標準').p(
        '智慧家庭中的家電耗能受居住活動、溫濕度與外部環境共同影響。可稽核的統計估計可協助使用者理解感測資訊與耗電量之關聯，也可作為資料分析教學案例。本研究不包含裝置即時串流、控制策略或節能介入實驗，因此不推論部署後可節省多少電力。').p(
        '研究問題一：在排除隨機變數與預設不可部署的 lights 後，哪些特徵方案能兼顧 Validation 誤差、CV 穩定性與可取得性？研究問題二：線性估計對尖峰耗電的限制為何？研究問題三：在時間相依資料下，如何誠實呈現區間不確定性？').p(
        '預測任務定義為 Contemporaneous Energy Consumption Estimation。輸入與 Appliances 來自同一時間窗；模型未建立 lag、預測視窗或未來氣象情境，因此不是 Forecast。What-If 預定介面只能展示模型關聯下的敏感度，不代表調整溫度會造成特定節能效果。').p(
        '成功標準以研究流程與可重現性為主：資料真實可追溯、時間切分不重疊、內外折無前處理／Target 洩漏、正式模型使用合法14-27特徵、版本與Hash凍結，以及圖表與數值可由來源重建。依PDD不設定虛構最低R²或RMSE改善門檻。低分或模型假設偏離仍是必須報告的研究結果。').p(
        '利害關係人包含課程助教、研究者與未來網站使用者。助教需要來源與程式對照，研究者需要每折稽核紀錄，使用者需要原始點估計、警告、區間限制與模型版本。正式Test成績不作為再次調整模型的觸發條件。')

    a.page('2 Data Understanding：資料來源與欄位').p(
        '資料名稱為 Appliances Energy Prediction。專案優先檢查 Kaggle；正式取得的 CSV 來自 UCI 官方封存檔，Kaggle對照紀錄另存，不宣稱不存在的Kaggle release編號。UCI說明資料來自低耗能住宅之感測與氣象觀測，並提供變數語意 [R1]。').table(
        'Dataset Summary',['項目','實際驗證值'],[['原始筆數',f"{quality['rows']:,}"],['原始欄位數',quality['columns']],
        ['時間起點',quality['time_start']],['時間終點',quality['time_end']],['取樣間隔','10分鐘；間隔分布見品質報告'],
        ['Target','Appliances，Wh'],['資料版本','以原始CSV SHA-256識別']],quality_path,'ALL_RAW',widths=[135,345]).table(
        '欄位分類與候選集合',['分類','欄位／用途','數量'],[['時間戳','date：排序及日曆衍生，不直接入模',1],
        ['能源欄位','Appliances為Target；lights為研究對照',2],['室內外感測','T1-T9、RH_1-RH_9',18],
        ['戶外氣象','T_out、Press_mm_hg、RH_out、Windspeed、Visibility、Tdewpoint',6],['隨機欄位','rv1、rv2：排除',2]],quality_path,'ALL_RAW',widths=[85,355,40]).p(
        '原始29欄含date與Target；移除rv1/rv2、排除date/Target，加入Hour/DayOfWeek/IsWeekend後有28個研究候選特徵；部署再排除lights，剩27個候選。這是候選集合差異，不表示正式模型使用全部27或28欄。')

    a.page('2.1 Data Understanding：品質與授權').table(
        '資料品質檢查摘要',['檢查','真實結果','處理原則'],[['缺失值總數',sum(quality['missing'].values()),'仍在Pipeline保留median Imputer'],
        ['重複完整列',quality['duplicate_rows'],'沒有任意刪列'],['重複時間戳',quality['duplicate_timestamps'],'保持唯一時間索引'],
        ['原始時間排序',str(quality['original_time_sorted']),'依時間順序切分，不Shuffle'],
        ['物理範圍違規總數',sum(quality['physical_range_violations'].values()),'品質規則通過不代表所有統計極端值皆異常'],
        ['候選特徵數',quality['candidate_count'],'研究含lights'],['部署候選數',quality['deployable_count'],'排除lights/rv1/rv2']],quality_path,'ALL_RAW',widths=[125,80,275]).p(
        '缺失值與物理範圍檢查結果來自真實執行，不以資料集說明頁代替品質驗證。耗電高峰可能是真實家電活動，不因其偏離平均值而刪除；本研究保留高耗電列，以免低估部署風險。品質報告亦保存欄位型別、描述統計與相鄰時間間隔分布。').p(
        'UCI官方頁標示CC BY 4.0 [R1, R9]。公開研究產品保留作者Luis Candanedo、Dataset DOI、來源及授權連結，並說明時間切分、特徵工程、模型與圖表轉換。原始下載時間與Archive Hash可查provenance.json；公開包不包含整份原始CSV。').p(
        '單一住宅與單一觀測期間限制外部效度。天氣與活動型態可能隨季節變動；資料中的相關性不能直接推論其他家庭或裝置。Wh是能源量，不等同瞬時功率W，亦未換算為電費或長期節能效益。')

    a.page('3 Data Preparation：時間切分與資訊邊界').table(
        '固定Chronological Split',['區段','筆數','起點','終點','用途'],[[n,m['rows'],m['start'],m['end'],
        {'train':'Fit與CV','validation':'已鎖候選比較','calibration':'只估計q','test':'一次凍結評估'}[n]]
        for n,m in split['partitions'].items()],split_path,'ALL_RAW',widths=[65,45,130,130,110]).p(
        '整數規則為floor(70%)、floor(10%)、floor(5%)，剩餘分配給Test。每區段保存CSV SHA-256、列索引範圍及不重疊證據。所有資料採來源所在地naive calendar，不依瀏覽器時區轉換。Hour為0-23，DayOfWeek以Monday=0、Sunday=6；Saturday/Sunday的IsWeekend=1。').p(
        '研究每個外折內重新Fit Imputer、Scaler、Feature Selector與Estimator；需要驗證誤差的選擇器另用外折訓練資料內的TimeSeriesSplit。不得先在完整Training Fit前處理再交叉驗證。Validation僅在名單鎖定後開啟；Calibration與Test不參與選模。').p(
        '選模完成後才合併Train+Validation，以已固定15欄重新Fit median Imputer、StandardScaler與LinearRegression。此時不重跑SFS。Calibration只產生絕對殘差；Test只讀凍結模型並預測。前處理統計與係數的來源因此與區間q的來源分開。')

    meanings=['廚房溫度','廚房相對濕度','洗衣房溫度','洗衣房相對濕度','辦公室溫度','浴室溫度','浴室相對濕度',
              '熨燙房溫度','青少年房間相對濕度','主臥溫度','主臥相對濕度','風速','能見度','露點溫度','日曆小時']
    a.page('3.1 Data Preparation：正式15特徵契約').table(
        '凍結特徵清單與順序',['順序','Feature','意義','單位'],[[i+1,f,meanings[i],model['features'][i]['unit']]
        for i,f in enumerate(model['feature_order'])],ARTIFACT,'TRAIN_VALIDATION',widths=[45,105,235,95]).p(
        '清單與B3 Decision Record完全一致。模型不使用lights、rv1、rv2；DayOfWeek與IsWeekend雖是候選，未進入最終15欄。外部輸入採固定欄名及單位，不猜測別名；缺欄是錯誤，顯式null才可用凍結median填補。NaN、Infinity、數字字串與布林值不能直接作為numeric JSON輸入。').p(
        '每筆輸入依Feature Order重排後，以z=(imputed_x-mean)/scale計算。Hour由固定date格式衍生，若另提供Hour須一致。缺失值規則可支援後續外部資料，但不表示正式資料具有缺失值；原始品質報告的缺失總數為零。')

    a.page('4 Modeling：Multiple Linear Regression').p(
        'Multiple Linear Regression以多個特徵的線性組合估計連續Target。Ordinary Least Squares最小化訓練資料的平方殘差和 [R3]。標準化後的推論式如下：').p(
        'z_j = (imputed_x_j - mean_j) / scale_j；predicted = intercept + sum(coefficient_j * z_j)，j=1,...,15。').p(
        f"正式截距為 {model['intercept']:.14f} Wh。所有median、mean、scale與係數只由Train+Validation取得。"
        '本模型不含交互作用、非線性基底或lag；因此難以表達家電開關造成的突發高耗能。標準化係數描述在其他特徵固定下的統計關聯，不應視為可實際介入的因果效果。').p(
        'Mean Baseline使用訓練區段的Target平均值；Full-Feature MLR使用全部候選；Selected-Feature MLR使用指定選擇器。Ridge以L2懲罰減少係數幅度，Lasso以L1懲罰產生稀疏係數，僅作研究比較。部署模型依PDD固定為sklearn.linear_model.LinearRegression。').p(
        '衡量預測誤差採MAE、MSE、RMSE與R²。令e=actual-predicted，MAE=mean(abs(e))，MSE=mean(e²)，RMSE=sqrt(MSE)。R²=1-SSE/SST；Adjusted R²=1-(1-R²)(n-1)/(n-p-1)，僅在Target變異有效且n>p+1時定義。Ridge/Lasso的傳統Adjusted R²不等同effective degrees of freedom估計。')
    a.table('Mean與Full-Feature MLR之Training CV',['基準模型','候選集合','CV RMSE','折間SD'],
        [[r.kind,r.candidate_set,fmt(r.cv_rmse_mean),fmt(r.cv_rmse_std)] for _,r in baselines.iterrows()],
        baseline_path,'TRAIN_CV',widths=[150,120,105,105],note='Wh；各5外折。Mean在兩集合數值等價；Full-MLR引用ALL canonical折，沒有新增Fit。這不是Test基準成績。')

    a.page('4.1 Modeling：五種Feature Selection').table(
        '五種方法的實際實作契約',['方法','本研究操作','資訊邊界'],[
        ['SelectKBest','f_regression單變量分數；取前K','外折Training內Fit前處理與分數'],
        ['RFE','LinearRegression、step=1遞迴刪除','只使用外折Training'],
        ['Forward SFS','逐步加入最小平均內折RMSE的特徵','內層TimeSeriesSplit=5；每候選重Fit前處理'],
        ['Lasso-based','標準化後alpha=1，依abs(coef)排序取K','保存非零係數數量；零係數同分依原欄序'],
        ['Permutation Importance','內折時間驗證區間，5次置換；平均重要性','只用外折Training內的5個時間子折']],ROOT/'src/features/selection.py','TRAIN_CV',widths=[110,185,185]).p(
        '一般選擇器概念可參照scikit-learn指南 [R6]；本研究的Forward SFS與Permutation由自訂TemporalFeatureSelector實作時間內折與Fit邊界稽核，而非直接採預設隨機CV。候選同分以原欄序決定，random seed=42。').p(
        'Lasso-based是exact-K排序比較，不保證所有被選欄位的Lasso係數非零，也不是只保留非零係數的SelectFromModel。K=ALL直接保留完整候選集合；數值等價的ALL折可引用canonical來源，明確記錄REUSED_ALL與零新增Fit，不假裝重新執行。')

    folds=csv(ROOT/'reports/evaluation/b2b/cv_fold_results.csv')
    cols=list(folds.columns)
    a.page('4.2 Modeling：TimeSeriesSplit與矩陣設計').p(
        '外層TimeSeriesSplit(n_splits=5)依時間擴增Training，後續區間作該折驗證 [R4]。內層SFS與Permutation同樣採5折時間切分。正式Training共13,814筆；每折獨立保存時間邊界、特徵清單、RMSE、Fit次數與Checkpoint Hash。').table(
        '正式研究矩陣',['維度','配置','數量'],[['Feature Selection','SelectKBest / RFE / SFS / Lasso / Permutation',5],
        ['候選集合','部署27／含lights研究28',2],['K','10 / 15 / 20 / 25 / ALL',5],['外折','TimeSeriesSplit',5],
        ['總計','50配置 × 5外折',250]],ROOT/'configs/b2b_matrix.json','TRAIN_CV',widths=[105,315,60]).p(
        'K10標記RESEARCH_ONLY；28集合是研究對照，不得作為預設部署模型。全矩陣含210個COMPUTED折與40個明確REUSED_ALL折，另有20個Mean/Full-MLR基準折。每折完成立即持久化，避免長時間執行中斷造成結果遺失。').p(
        'CV穩定性使用五折RMSE的population SD(ddof=0)；特徵穩定性使用折間Jaccard交集／聯集比例平均。這些是研究摘要，不是統計顯著性檢定。時間折gap=0，避免未來資訊洩漏不等同使相鄰殘差獨立；剩餘時間相依仍須在Evaluation揭露。')

    fixed=cv[(cv.candidate_set=='deployable27')&(cv.k.astype(str)=='15')]
    a.page('4.3 Modeling：同條件Feature Selection比較').table(
        '27候選／K15之Training CV',['方法','RMSE mean','折間SD','Jaccard','計算秒'],[[r.method,fmt(r.cv_rmse_mean),fmt(r.cv_rmse_std),fmt(r.feature_jaccard_mean),fmt(r.compute_seconds)]
        for _,r in fixed.iterrows()],cv_path,'TRAIN_CV',widths=[140,85,85,85,85],note='RMSE/SD單位Wh；每列5外折。成本為已保存計算成本，不是獨立重跑Benchmark。').figure(
        'Feature Selection與K配置的CV RMSE',b2fig/'feature_selection_rmse.png',cv_path,'TRAIN_CV',height=245).p(
        '固定27候選／K15時，RFE的CV RMSE最低；SFS並非Training CV最佳方法。因此正式採SFS的理由必須來自事前鎖定的Validation比較，不能倒推成所有研究條件下最優。相關特徵也可能使不同外折選到不同欄位。')

    paired_path=ROOT/'reports/evaluation/b2b/candidate_paired_comparison.csv'
    paired=csv(paired_path)
    krows=[]
    for k in ['10','15','20','25','ALL']:
        values=[]
        for candidate in ['deployable27','research28']:
            values.append(fmt(cv[(cv.candidate_set==candidate)&(cv.k.astype(str)==k)].cv_rmse_mean.mean()))
        krows.append([k,*values,'研究對照' if k=='10' else '依27集合部署規則'])
    a.page('4.4 Modeling：K值及27／28候選集合').table(
        '各K跨五方法的描述性平均CV RMSE',['K','27集合均值','28集合均值','適用性'],krows,cv_path,'TRAIN_CV',widths=[55,110,110,205],
        note='Wh；同K對五方法的CV均值再平均，並非單一模型分數；ALL在每個集合內為等價數值。').figure(
        '27／28集合的K曲線（不含ALL）',PRIVATE/'figures/training_cv.png',cv_path,'TRAIN_CV',height=255).p(
        '小K有時得到較低CV誤差，但K10不符合正式特徵政策。加入lights未呈現所有方法與K都一致改善；集合平均不能證明某一單獨特徵的因果效應。完整50列含ALL的數值保存於feature_selection_results.csv。')

    a.page('4.5 Modeling：候選集合與穩定性').figure(
        '同配置27／28候選集合對照',b2fig/'candidate_comparison.png',paired_path,'TRAIN_CV',height=210).figure(
        '不同配置的特徵穩定性',b2fig/'feature_stability.png',cv_path,'TRAIN_CV',height=210).p(
        'Jaccard高表示外折特徵清單較一致，不保證預測誤差最低，也不證明特徵重要性為因果。ALL的Jaccard=1反映保留全部欄位的定義；不能以此宣稱ALL必然具有最佳泛化能力。圖中研究28與部署27保留各自集合識別。')

    a.page('4.6 Modeling：Validation前鎖定候選').table(
        '事前鎖定四個合法MLR方案',['方案','K','CV RMSE±SD','Jaccard'],[[r['method'],r['k'],fmt(r['cv_rmse_mean'])+' ± '+fmt(r['cv_rmse_std']),fmt(r['feature_jaccard_mean'])]
        for r in lock['candidates']],lock_path,'TRAIN_CV',widths=[120,50,190,120]).p(
        '先限27候選與K15/20/25/ALL，以CV RMSE、SD、Jaccard及成本的固定排序選前三個非ALL，再強制加入Full27基準。不可覆寫鎖定紀錄早於Validation Target開啟；未依Validation成績新增候選、改K或修改規則。').p(
        '四方案各比較MLR、Ridge、Lasso，共12次Validation推論。Ridge alpha網格為0.01/0.1/1/10/100/1000；Lasso為0.01/0.1/1/10/100。alpha只以Training內部五折RMSE選擇；每折重做該方案選擇，不能把完整Training選定欄位帶入內折驗證。').p(
        '四方案均選Ridge alpha=1000、Lasso alpha=1。Ridge落在預先網格上界是搜尋限制，並未事後擴大網格追求更好的Validation分數。Ridge/Lasso保留研究身分，不自動取代正式MLR。').figure(
        '固定候選之Validation RMSE',ROOT/'reports/evaluation/b3/validation_rmse.png',val_path,'VALIDATION',height=215)

    a.page('4.7 Modeling：Validation MLR／Ridge／Lasso').table(
        '12組Validation實際指標',['方案','模型/α','MAE','MSE','RMSE','R²','Adj.R²'],[[str(r.method)+'/'+str(r.k),r.model+('/'+str(r.alpha) if str(r.alpha) else ''),
            fmt(r.mae),fmt(r.mse),fmt(r.rmse),fmt(r.r2,5),fmt(r.adjusted_r2,5)] for _,r in val.iterrows()],val_path,'VALIDATION',widths=[110,70,60,80,60,50,50],
        note='MAE/RMSE單位Wh；MSE為Wh²。Ridge/Lasso Adjusted R²採傳統特徵數，非有效自由度。').p(
        f"四個已鎖定MLR中，Forward SFS/K15的Validation RMSE={fmt(decision['validation_metrics']['rmse'],6)} Wh最低，"
        f"MAE={fmt(decision['validation_metrics']['mae'],6)} Wh、MSE={fmt(decision['validation_metrics']['mse'],6)} Wh²。"
        '依既定選模規則，正式保留LinearRegression及15欄方案；CV穩定性、特徵數與成本作預定同分排序依據。').p(
        'SFS的Lasso結果接近MLR，不足以宣稱顯著優劣；不同候選之差異亦未進行獨立統計顯著性推論。正式模型的決策來自有限Validation比較，不是從Test成績反向挑選。')

    a.page('4.8 Modeling：Final Fit與凍結係數').table(
        '正式模型係數（標準化座標）',['順序','Feature','Coefficient'],[[i+1,f,fmt(model['coefficients'][i],8)] for i,f in enumerate(model['feature_order'])],
        ARTIFACT,'TRAIN_VALIDATION',widths=[55,150,275],note=f"Intercept={model['intercept']:.14f} Wh；完整精度與median/mean/scale在model.json。")
    a.p('Final Fit只使用Train+Validation的15,787筆，2016-01-11 17:00:00至2016-04-30 08:00:00。Imputer、Scaler與Estimator各Fit一次；SFS不再執行。Calibration與Test沒有更新前處理或係數。版本mlr-1.0.0與Artifact Hash在校準及評估前後保持凍結。')

    selected_cv=cv[(cv.candidate_set=='deployable27')&(cv.method=='forward_sfs')&(cv.k.astype(str)=='15')].iloc[0]
    a.page('5 Evaluation：分區段讀取模型表現').table(
        '正式方案不同階段之誤差',['區段','模型狀態','RMSE(Wh)','R²','解讀'],[
        ['Training CV','每折重新選擇/訓練',fmt(selected_cv.cv_rmse_mean),fmt(selected_cv.cv_r2_mean,5),'五折均值，非單一Final模型'],
        ['Validation','完整Train Fit',fmt(decision['validation_metrics']['rmse']),fmt(decision['validation_metrics']['r2'],5),'有限候選選模'],
        ['Test','Train+Validation凍結Fit',fmt(metrics['rmse']),fmt(metrics['r2'],5),'一次正式評估']],PUBLIC/'research_summary.json','各列明確標示',widths=[80,140,80,70,110]).table(
        '正式Test指標',['MAE(Wh)','MSE(Wh²)','RMSE(Wh)','R²','Adjusted R²'],[[fmt(metrics[k],6) for k in ['mae','mse','rmse','r2','adjusted_r2']]],
        test_path,'TEST',widths=[90,115,100,85,90]).figure(
        'Test Actual vs Predicted',c2/'figures/prediction_vs_actual.png',c2/'predictions.csv','TEST',height=240).p(
        'Test R²約0.078，顯示相對於Test樣本平均值僅有有限的平方誤差解釋能力，不應描述為高準確度。Mean Baseline與Full-MLR基準僅有Training CV比較，不補造其Test成績。Validation與Test的模型Fit資料不同，不能把分數差異直接當成改善。')

    a.page('5.1 Evaluation：殘差與高耗電尖峰').figure(
        'Test Residual vs Predicted',c2/'figures/residual_vs_predicted.png',c2/'predictions.csv','TEST',height=215).figure(
        'Test Residual Distribution',c2/'figures/residual_distribution.png',c2/'predictions.csv','TEST',height=215).p(
        '殘差定義為actual-predicted。右側長尾與明顯正殘差反映高耗電峰值常被低估，線性組合無法充分捕捉家電開關行為。RMSE對大殘差敏感，因而明顯高於MAE；這是模型限制，不應刪除高峰後重新包裝成績。')

    a.page('5.2 Evaluation：常態性與時間相依').figure(
        'Test Residual Q-Q Plot',c2/'figures/residual_qq.png',c2/'predictions.csv','TEST',height=210).figure(
        'Test Residual ACF',c2/'figures/residual_acf.png',diag_path,'TEST',height=210).p(
        f"Q-Q右尾大幅偏離常態參考線。DW={fmt(diag['durbin_watson'],6)}、lag-1 ACF={fmt(diag['residual_acf'][1],6)}，"
        '顯示殘差有明顯正向時間相依；不能宣稱殘差獨立或線性模型統計推論假設皆成立。ACF為描述性診斷，圖上未附經調整後的假設檢定。')

    a.page('5.3 Evaluation：95% Split Conformal方法').p(
        '使用完全獨立的Calibration計算score_i=abs(actual_i-predicted_i)。有限樣本校正採k=ceil((n+1)*(1-alpha))，q為排序後第k個殘差，不採線性插值分位數 [R5]。若k>n則不能產生有限q，應明確報錯。').table(
        '凍結Conformal Calibration',['參數','實際值','來源'],[['n',cal['n'],'Calibration有效殘差'],['alpha',cal['alpha'],'預定名目5%誤覆蓋率'],
        ['k',cal['k'],'ceil((986+1)*0.95)'],['q(Wh)',repr(cal['q']),'第938個排序絕對殘差']],cal_path,'CALIBRATION',widths=[75,190,215]).p(
        '固定區間：Lower=max(0,predicted-q)，Upper=predicted+q。Point Prediction不裁切；負值保留並加NEGATIVE_ESTIMATE_WARNING。Upper<Lower時加INVALID_INTERVAL，不調換端點或假裝有效。Conformal區間是單筆耗電估計不確定性，不是平均反應的參數信賴區間。').figure(
        'Test凍結Prediction Interval',c2/'figures/prediction_interval.png',c2/'predictions.csv','TEST',height=225).p(
        'Split Conformal之有限樣本覆蓋結論依賴交換性。Chronological資料受時間相依與分布漂移影響，因此本研究不能把名目95%轉述為未來時間區段的嚴格保證；非負下界裁切也需揭露。')

    a.page('5.4 Evaluation：Coverage Audit與一次性評估').table(
        '實際Test區間稽核',['項目','實際值'],[['全部Test筆數',pi['total_rows']],['有效／無效區間',f"{pi['valid_intervals']} / {pi['invalid_intervals']}"],
        ['涵蓋筆數',pi['covered_count']],['全體／有效Coverage',f"{100*pi['coverage_all']:.4f}% / {100*pi['coverage_valid']:.4f}%"],
        ['Average Width(Wh)',fmt(pi['average_width_valid'],6)],['Median Width(Wh)',fmt(pi['median_width_valid'],6)],
        ['Below Lower／Above Upper',f"{pi['below_lower_count_valid']} / {pi['above_upper_count_valid']}"],
        ['負值預測／Warning',f"{pi['negative_prediction_count']} / {pi['negative_warning_count']}"],['Invalid Warning',pi['invalid_warning_count']]],
        test_path,'TEST',widths=[235,245]).figure(
        '全部Test時間序列預測',c2/'figures/time_series_prediction.png',c2/'predictions.csv','TEST',height=200).p(
        'Run ID為c2-mlr-1.0.0-test-v1。首次紀錄先於Test Target解析，正式pipeline.predict只執行一次，模型Fit次數為零。全部2,962列保存timestamp、actual、prediction、residual、absolute error與區間；後續報告只讀保存產物。若有無效區間，全體Coverage仍須把它們算為未涵蓋，不能只報排除後比例。')

    a.page('5.5 Evaluation：BP與VIF統計診斷').table(
        'Test殘差診斷摘要',['診斷','數值','限制'],[['Durbin-Watson',fmt(diag['durbin_watson'],6),'序列殘差明顯相依'],
        ['Koenker BP LM',fmt(bp['lm_statistic'],6),'robust=True；含截距與15特徵'],['BP LM p',f"{bp['lm_pvalue']:.6e}",'名目p值受序列相依限制'],
        ['BP F／p',fmt(bp['f_statistic'],6)+' / '+f"{bp['f_pvalue']:.3e}",'不能據此宣稱同方差'],
        ['Q-Q／殘差分布','右尾偏離常態','描述性圖；未宣稱常態性通過']],diag_path,'TEST',widths=[100,190,190]).table(
        '15特徵VIF依資料來源分開',['Feature','Train+Validation VIF','Test VIF'],[[x['feature'],fmt(x['vif']),fmt(y['vif'])] for x,y in zip(vif['vif'],diag['vif'])],
        PUBLIC/'research_summary.json','TRAIN_VALIDATION / TEST',widths=[140,170,170],note='兩設計均加截距、15特徵；rank16/16。Train+Validation資料來源為C3 VIF，Test來源為C2。').p(
        'BP採Koenker robust版本 [R7]，VIF用各特徵對其他欄位的輔助回歸 [R8]。輔助診斷回歸不等於重新訓練正式預測模型。高VIF提示係數可能不穩定，不是移除特徵的事後授權，也不提供因果解釋。')

    a.page('5.6 Evaluation：研究限制與可用範圍').figure(
        'Train+Validation與Test VIF對照',PRIVATE/'figures/vif_by_split.png',PUBLIC/'research_summary.json','TRAIN_VALIDATION / TEST',height=250).p(
        f"Train+Validation的最高VIF為T9={fmt(max(vif['vif'],key=lambda r:r['vif'])['vif'])}；Test最高為RH_1={fmt(max(diag['vif'],key=lambda r:r['vif'])['vif'])}。"
        '設計矩陣的共線性隨區段改變，不能把其中一個值當成全資料固定性質。').p(
        '限制包括：單住宅外部效度有限；同時段感測器必須可取得；線性模型無法描述所有家電活動；共線性限制係數解釋；時間相依限制統計檢定與Conformal交換性；固定區間寬度無法完全反映不同耗電情境的異質誤差。').p(
        '本研究保留所有原始高耗電列與43個負點估計，不以裁切點估計美化誤差。模型可作教學與關聯分析，不能保證未來尖峰、節能收益或安全控制效果。任何未來改模型、加入lag或採用非線性方法都需要新的研究版本與獨立評估設計，不能再以既有Test調參。')

    a.page('6 Deployment：已完成產物與尚未實作功能').table(
        '部署狀態與責任邊界',['項目','目前狀態','說明'],[['Python Pipeline','已完成','只供離線重現，不上傳Pages'],
        ['JSON Model Artifact','已完成','15欄、係數、前處理、q及Warning契約'],['Research Summary／Charts','已完成','16公開JSON；首頁摘要28,491bytes'],
        ['Python Reference／Fixtures','已完成','JS parity仍NOT_RUN'],['五頁Dashboard','尚未實作','不宣稱已有互動網站'],
        ['What-If／External CSV','只有契約','待前端實作與安全／效能驗證'],['GitHub Pages','尚未部署','待授權及部署QA']],PUBLIC/'manifest.json','部署狀態',widths=[145,85,250]).p(
        '規劃架構為Python離線研究、Artifact匯出、靜態HTML/CSS/JavaScript與Browser Inference。Pages只發布公開allowlist，瀏覽器用JSON固定參數執行median填補、標準化、線性估計與區間；不需要Python Backend或joblib。').p(
        '預定五頁包括Overview、Data & Features、Model Evaluation、What-If Simulator與External CSV Prediction。外部CSV採HTTPS URL或本機上傳、固定date與15特徵契約、明確錯誤格式；初始10MB／20,000rows限制與CORS替代流程均屬待實作規劃，尚無瀏覽器完成證據。')

    a.page('6.1 Deployment：維護、重現與結論').p(
        '維護時先核對model_version、schema_version與SHA-256，再允許推論。上傳資料不應回傳伺服器或覆寫官方Test指標；每筆結果應保留版本、原始點估計及Warning。缺欄、日期不合法、非有限數或版本不符應阻斷輸出，不能改用虛構預測。').p(
        '初始研究摘要與attribution僅28,491bytes；圖表按需載入495點小時彙整、446點標記取樣的scatter及完整筆數histogram。取樣不代表公平統計樣本，hourly endpoint mean也不是對hourly average的95%校準區間。官方指標始終來自原始保存結果。').p(
        '重現分為「產物驗證／報告重建」與「隔離工作副本的完整研究重跑」。前者不讀Dataset或載入模型，後者會重新訓練，必須在另建工作副本依固定程序執行。正式Test已完成，任何重跑只能作重現驗證，不得重新當作模型選擇依據。詳細命令與助教對照表見REPRODUCIBILITY_README.md。').p(
        '結論：本研究完成可追溯的線性估計實驗、合法15特徵模型及獨立區間校準。Test結果與殘差診斷指出模型對高耗電尖峰與時間相依仍有明顯限制；觀察到約95.68%的Test Coverage不構成未來保證。研究流程與誠實揭露是主要交付價值。網站與NotebookLM／AI對話交付尚不屬本階段完成範圍。').table(
        '本次文件工具版本',['套件','實際版本'],[[n,version(n)] for n in ['reportlab','pypdf','scikit-learn','statsmodels','numpy','pandas']],
        ROOT/'reports/environment/requirements-m2a-311-lock.txt','環境紀錄',widths=[230,250],note='ML套件版本沿用；PDF套件僅加在專案.venv，另存M2-A實際lock。')

    for section,group in [('參考文獻（一）',refs[:5]),('參考文獻（二）',refs[5:])]:
        a.page(section)
        for ref in group:
            a.p(f"[{ref['id']}] {ref['citation']}").p(ref['url']).p('用途：'+ref['supports']+' 核實日期：'+ref['verified']+'。')
        if section.endswith('（二）'):
            a.p('專案內部來源：Final PDD v1.0；Part A品質與split manifest；B2-B研究CSV與checkpoint；B3鎖定與Validation紀錄；C1正式模型與Calibration；C2保存Test預測與診斷；C3Research Summary、VIF與公開資料Manifest。內部來源不是外部文獻，所有Hash及圖表逐項索引另附。')

    a.page('附錄：助教檢查與模型來源對照').table(
        '六階段程式與研究產物對照',['工作','程式模組','主要證據'],[
        ['資料取得／準備','src/data/acquire.py；prepare.py','reports/data/provenance.json；quality_report.json'],
        ['CV與選擇','src/features/selection.py；training/matrix.py','reports/evaluation/b2b/feature_selection_results.csv'],
        ['Validation選模','src/training/b3.py','reports/experiments/b3-validation-v1/mlr_decision.json'],
        ['Final Fit／Calibration','src/training/c1.py','models/offline/mlr-1.0.0/；C1 manifest'],
        ['一次Test／診斷','src/evaluation/c2.py；c2_report.py','reports/evaluation/c2-mlr-1.0.0/'],
        ['公開契約','src/export/c3.py；c3_validate.py','artifacts/deployment/mlr-1.0.0/'],
        ['本報告','src/export/academic_report.py','Markdown／PDF／source index／report manifest']],ROOT/'docs/pdd/Final_PDD_v1.0.md','研究重現',widths=[100,170,210]).p(
        '報告由單一結構化來源同時產生Markdown與PDF，表格數字讀自真實CSV/JSON，圖像沿用既有研究產物。顯示小數僅為版面格式，不修改原始指標；fact ledger保存原始值與顯示值。可重算性測試核對來源值、模型Hash、圖表區段、PDF內數字及中文字型嵌入。').p(
        '來源索引包含每張圖、每張表的編號、資料區段、原檔與SHA-256；Manifest包含報告、原稿、索引、程式、字型Hash及環境版本。PDF頁碼、中文排版與圖像另以渲染PNG人工視覺核對。').p(
        '本階段僅完成M2-A文件；不宣稱整份作業已ASSIGNMENT_READY。NotebookLM、真實AI對話證據及網站／部署驗收仍需依後續授權與階段要求處理。')
    return a


def markdown(a):
    lines=[]
    for page in a.pages:
        lines.extend(['# '+page['title'],''])
        for b in page['blocks']:
            if b['kind']=='paragraph':lines.extend([b['text'],''])
            elif b['kind']=='table':
                lines+=['**'+b['id']+' '+b['title']+'**','', '| '+' | '.join(map(str,b['headers']))+' |','|'+'|'.join(['---']*len(b['headers']))+'|']
                lines += ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in b['rows']]
                lines+=['',f"資料區段：{b['split']}；來源：`{b['source']}`。{b['note']}",'']
            else:
                image=Path(b['path']);relative='../../'+str(image).replace('\\','/')
                lines += [f"![{b['id']} {b['title']}]({relative})",'',f"{b['id']} {b['title']}。資料區段：{b['split']}；來源：`{b['source']}`。{b['note']}",'']
        lines+=['<!-- PAGE -->','']
    return '\n'.join(lines)


def render(a,path):
    pdfmetrics.registerFont(TTFont('TC',str(FONT),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('TCBold',str(BOLD),subfontIndex=0))
    styles={
        'body':ParagraphStyle('Body',fontName='TC',fontSize=10.2,leading=16.3,spaceAfter=10,wordWrap='CJK'),
        'title':ParagraphStyle('Title',fontName='TCBold',fontSize=18,leading=24,spaceAfter=18,wordWrap='CJK',textColor=colors.HexColor('#143449')),
        'small':ParagraphStyle('Small',fontName='TC',fontSize=7.8,leading=11.4,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor('#485760')),
        'cell':ParagraphStyle('Cell',fontName='TC',fontSize=8.2,leading=11.6,wordWrap='CJK'),
        'header':ParagraphStyle('Header',fontName='TCBold',fontSize=8.2,leading=11.6,wordWrap='CJK',textColor=colors.white)}
    story=[]
    def p(text,style='body'):return Paragraph(escape(str(text)),styles[style])
    for page in a.pages:
        story.append(p(page['title'],'title'))
        for b in page['blocks']:
            if b['kind']=='paragraph':story.append(p(b['text']))
            elif b['kind']=='table':
                story.append(p(b['id']+' '+b['title'],'small'))
                data=[[p(x,'header') for x in b['headers']]]+[[p(v,'cell') for v in row] for row in b['rows']]
                table=Table(data,colWidths=b['widths'],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#244b60')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f2f6f8'),colors.white]),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                    ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#244b60'))]))
                story.append(table);story.append(Spacer(1,5))
                story.append(p(f"區段：{b['split']}；來源：{b['source']}。{b['note']}",'small'))
            else:
                image=Image(str(ROOT/b['path']))
                factor=min(480/image.imageWidth,b['height']/image.imageHeight)
                image.drawWidth=image.imageWidth*factor;image.drawHeight=image.imageHeight*factor
                image.hAlign='CENTER'
                caption=p(f"{b['id']} {b['title']} | {b['split']} | 來源：{b['source']}。{b['note']}",'small')
                story.append(KeepTogether([image,Spacer(1,5),caption]))
        story.append(PageBreak())
    def chrome(canvas,doc):
        canvas.saveState();canvas.setFont('TC',8);canvas.setFillColor(colors.HexColor('#657681'))
        canvas.drawString(20*mm,A4[1]-13*mm,'SMART HOME ENERGY | CRISP-DM | mlr-1.0.0')
        canvas.line(20*mm,15*mm,A4[0]-20*mm,15*mm)
        canvas.drawString(20*mm,10*mm,'研究文件 M2-A / 2026-10-17')
        canvas.drawRightString(A4[0]-20*mm,10*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,
        topMargin=22*mm,bottomMargin=22*mm,title='智慧家庭能源消耗估計與分析 - CRISP-DM研究報告',author='Smart Home Energy Research Project')
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)


def build():
    with C1Gate('none') as gate:
        a=content()
        OUT.mkdir(parents=True,exist_ok=True);PDF.parent.mkdir(parents=True,exist_ok=True);MD.parent.mkdir(parents=True,exist_ok=True)
        MD.write_text(markdown(a),encoding='utf-8')
        render(a,PDF)
        reader=PdfReader(PDF)
        index=dict(figures=a.figures,tables=a.tables,inputs=a.inputs)
        (OUT/'source_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
        index_lines=['# 圖表與表格來源索引','', '| 編號 | 標題 | 資料區段 | 來源 | SHA-256 |','|---|---|---|---|---|']
        for b in [*a.tables,*a.figures]:index_lines.append(f"| {b['id']} | {b['title']} | {b['split']} | {b['source']} | {b['source_sha256']} |")
        (ROOT/'docs/research/FIGURE_TABLE_SOURCE_INDEX.md').write_text('\n'.join(index_lines)+'\n',encoding='utf-8')
        (OUT/'fact_ledger.json').write_text(json.dumps(a.facts,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'document_structure.json').write_text(json.dumps(a.pages,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest=dict(status='GENERATED_PENDING_QA',at_utc=now(),model_version='mlr-1.0.0',pages=len(reader.pages),logical_pages=len(a.pages),
            figures=len(a.figures),tables=len(a.tables),inputs=a.inputs,fonts={str(p):sha(p) for p in [FONT,BOLD]},
            packages={n:version(n) for n in ['reportlab','pypdf','scikit-learn','statsmodels']},
            source_dataset_reads=gate.reads,model_fit_calls=0,model_prediction_calls=0,
            files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                [PDF,MD,OUT/'source_index.json',OUT/'fact_ledger.json',OUT/'document_structure.json',ROOT/'docs/research/FIGURE_TABLE_SOURCE_INDEX.md',Path(__file__)]})
        (OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in manifest.items() if k not in ['inputs','files','fonts']},ensure_ascii=False,indent=2))
        return manifest


if __name__=='__main__':build()
