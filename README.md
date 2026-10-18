# Smart Home Energy Consumption Prediction & Analytics

> 課程機器學習專題｜CRISP-DM｜Multiple Linear Regression｜純靜態互動式 Dashboard

**Planned Demo — Not Yet Deployed:**  
https://aidenxiao0924.github.io/HW2-Smart-Home-Energy-Prediction/

目前網址僅為規劃目標；公開 Repository 尚無首次提交，GitHub Pages 尚未啟用或完成公開驗收。

## Project Overview

本專題使用 Appliances Energy Prediction 資料，估計同一時間區間的家電耗電量 `Appliances`（Wh）。這是依同時段感測器與氣象資料進行的能源消耗量估計，不是未來時間序列 Forecast。

課程交付採 CRISP-DM 六階段：Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation、Deployment。正式模型為 `mlr-1.0.0`，使用 `sklearn.linear_model.LinearRegression`、Forward SFS 在研究階段凍結的15個特徵，以及獨立Split Conformal校準的Prediction Interval。

## Course & Assignment Information

- 作業：HW2 — Smart Home Energy Consumption Prediction & Analytics
- 方法：CRISP-DM、時間順序切分、TimeSeriesSplit
- 執行環境：Python 3.11、scikit-learn；網站為HTML／CSS／Vanilla JavaScript
- 公開狀態：Deployment preflight；尚未上線

## Dataset

- 來源：Kaggle Appliances Energy Prediction索引／UCI Machine Learning Repository官方資料；本研究正式CSV取自UCI可追溯來源
- 作者：Luis Candanedo 等
- 授權：CC BY 4.0
- 19,735筆、29個原始欄位、10分鐘取樣
- 正式研究採時間順序切分：Train 70%、Validation 10%、Calibration 5%、Test 15%

完整原始Dataset不納入Public Repository；取得方式、Hash與授權紀錄請見 `docs/public/`。

## Modeling與Feature Selection

研究矩陣比較五種Feature Selection：SelectKBest、RFE、Forward Sequential Feature Selection、Lasso-based Selection、Permutation Importance；候選集合為27個部署候選及含`lights`的28個研究對照，K為10、15、20、25、ALL，共50配置／250個外折。

正式模型使用Forward SFS、K=15：

`T1, RH_1, T3, RH_3, T4, T5, RH_5, T7, RH_8, T9, RH_9, Windspeed, Visibility, Tdewpoint, Hour`

## Frozen Evaluation Results

| 指標 | 正式Test結果 |
|---|---:|
| MAE | 48.138654 Wh |
| MSE | 7,613.538875 Wh² |
| RMSE | 87.255595 Wh |
| R² | 0.077777 |
| Adjusted R² | 0.073081 |
| Observed PI Coverage | 95.6786% |

Test只進行一次正式凍結評估，沒有依Test結果重新選模。R²偏低，代表線性模型的解釋能力有限；高耗電峰值常被低估。

## Prediction Interval

獨立Calibration Set以Split Conformal計算名目95% Prediction Interval，凍結 `q = 185.3571214361994 Wh`：

- `Lower = max(0, Prediction - q)`
- `Upper = Prediction + q`

時間序列不保證交換性；Observed Test Coverage不代表每筆或未來資料都有嚴格95%涵蓋保證。

## Dashboard Features

1. Overview：研究背景、KPI、時間切分與限制。
2. Data & Features：Feature Selection、Training Distribution、Pearson Heatmap與獨立Importance。
3. Model Evaluation：Training CV、Validation、Test、Prediction Interval與殘差診斷。
4. What-If Simulator：瀏覽器Frozen Engine推論、PI與逐特徵Training Min/Max提示。
5. External CSV Prediction：本機上傳或HTTPS URL、批次推論、外部指標與安全CSV匯出。

OOD Min／Max只檢查每個特徵是否超出原始Training單變量範圍，不是完整的多變量OOD Detector，也不會裁切或阻擋預測。外部CSV的準確度取決於外部資料品質，且只有提供合法`Appliances`時才能計算指標。

Test Demo展示12筆按固定時間等距規則選出的可追溯Test樣本；它不是新的正式Test Evaluation，也不代表整體模型表現。

## Proposed Public Project Structure

```text
├─ README.md
├─ requirements.txt
├─ package.json
├─ src/                         # Python資料與ML程式
├─ configs/                     # 可重現實驗設定
├─ tools/                       # 匯出、驗證與RC工具
├─ tests/                       # Python/JavaScript/E2E測試來源
├─ web/                         # 靜態網站原始碼
├─ artifacts/deployment/        # 核准公開的Model JSON、Schema與聚合研究資料
├─ artifacts/training/train-v1/ # TRAIN聚合統計，不含逐列資料
├─ artifacts/demo/test-v1/      # 12筆可追溯Demo最小資料
├─ releases/web-rc3/            # 凍結公開網站與Manifest
└─ docs/                        # 研究、授權、部署與截圖文件
```

原始／切分Dataset、`.venv`、瀏覽器快取、`.joblib`、逐列內部研究結果、原始對話與私人文件不公開。

## Installation & Reproduction

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
node --test tests/javascript/*.test.mjs
python -m http.server 8000 --directory releases/web-rc3/site
```

Dataset須依重現文件從正式來源取得並核對SHA-256。正式Test已完成；重現結果不得用來重新選模或修改凍結Test成績。

## Technologies

Python 3.11、pandas、NumPy、scikit-learn、SciPy、statsmodels、Matplotlib、pytest、HTML、CSS、Vanilla JavaScript、SVG、Node.js node:test、Playwright、GitHub Pages（規劃中）。

## Limitations & Academic Notes

- 同期耗電量估計，不是未來Forecast。
- Test R²約0.077777，模型解釋能力有限。
- Prediction Interval為名目95%，不保證未來嚴格覆蓋。
- Training Min/Max只提供逐特徵警告。
- Correlation與Permutation Importance不代表因果。
- 網站、外部CSV與What-If結果不可解讀為節能因果效果。
- Planned Demo在M4-B2完成真實HTTPS與三瀏覽器驗收後，才可改標Live。


## License

本專案原始碼與本專案撰寫的文件採 MIT License。UCI Dataset 仍依 CC BY 4.0 授權；第三方套件、字型與其他另行歸屬的內容不因本專案 MIT License 而改變其原授權。

