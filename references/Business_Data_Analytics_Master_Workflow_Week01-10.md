# Business Data Analytics — Master Workflow (Week 01–10)

**ฐานเอกสาร:** 961731 Business Data Analytics จากไฟล์ที่แนบในโปรเจกต์: Week 01, 02, 03, 04, 05, Week 06 (Enterprise CLV + Case Study 2), Week 08, 09, 10; **ไม่มีไฟล์ Week 07 ในชุดที่ตรวจ**

**แนวคิดแกนกลาง:** Business Objective → Data Understanding / EDA → Data Preparation & Temporal Integrity → Right Analytical Method → Reliable Evaluation → Interpretable Business Action → Financial Impact → Experiment & Monitoring.

## A. Workflow กลางก่อนเริ่มทุกงาน (CRISP-DM)

1. **Business Understanding:** นิยามปัญหาให้ชัด เช่น ยอดขายลดลง, สินค้าคู่ใดควรจัด Bundle, ลูกค้ากลุ่มใดควรได้รับงบการตลาด, ใครเสี่ยง Churn หรือสังคมบ่นเรื่องอะไร กำหนด KPI, baseline, ช่วงเวลา, กลุ่มเป้าหมาย, เงื่อนไขการตัดสินใจ, ต้นทุน และการกระทำที่เป็นไปได้
2. **Data Understanding & Audit:** ระบุหน่วยวิเคราะห์ (transaction, customer, customer-time snapshot, invoice, post, process event) และ data dictionary ตรวจ schema/dtype/unique keys/missing/duplicates/anomalies/returns/currency/units/time range ความครบถ้วน ความถูกต้อง ความสอดคล้อง และความทันเวลา
3. **EDA / Hypothesis-led Analysis:** ตรวจ distribution, mean/median/mode, standard deviation/IQR/skew/outliers, categorical frequencies, cross-tab/chi-square ภายใต้เงื่อนไข, group comparison, timeline/trends, graphical patterns และ data bias; ใช้ MECE issue tree/value-driver decomposition เพื่อตอบสมมติฐานธุรกิจ ไม่ไล่หา correlation แบบไร้เป้าหมาย
4. **Preparation & Feature Engineering:** ทำ cleaning แบบมีเหตุผล เก็บ audit trail; แยกเวลาและนิยาม as-of date ก่อนสร้าง features, แยก train/validation/test ตามหน่วยและเวลาให้ตรง task; fit scalers, encoders, imputers, vectorizers, resamplers ภายใน training folds เท่านั้น ป้องกัน target/time/preprocessing/group leakage
5. **Methods & Modeling:** เลือก descriptive, association, clustering, probabilistic CLV, regression, classification หรือ NLP ให้ตรงโจทย์ พร้อม baseline เปรียบเทียบและหลักฐานความเหมาะสม
6. **Evaluation & Interpretation:** ประเมิน metrics ที่สอดคล้องเป้าหมาย บน holdout ที่ไม่ปน training ตรวจความเสถียร calibration (ถ้าคะแนนเป็น probability), subgroup errors, assumptions, explainability และความหมายเชิงธุรกิจ ห้ามเอาความสัมพันธ์หรือ SHAP มาแสดงเป็น causal effect
7. **Decision & Finance:** เปลี่ยน insight/model score เป็น Type 1 (ต้นทุนสูง/กลับยาก ต้องการหลักฐานมาก) หรือ Type 2 (ทดลองปรับง่าย) แล้วคำนวณ expected incremental profit = incremental revenue × contribution margin − campaign/operation cost − cannibalization/other costs; แยก observation, forecast, assumptions และ verified causal lift
8. **Experiment & Monitor:** วาง A/B test/holdout, KPI, horizon, stop rule; ตรวจผลจริงและติดตาม drift, data quality, business value, costs, fairness และ latency แล้ววนปรับ

## B. Decision Routing: ถ้า Dataset/Problem เป็นแบบนี้ ใช้อะไร

| ปัญหา / ลักษณะข้อมูล | วิธีจากสไลด์ | สิ่งที่ต้องเตรียม | ผลลัพธ์และ Action |
|---|---|---|---|
| อยากรู้ว่า KPI ผิดปกติอะไร / ยอดขาย ต้นทุน เวลาให้บริการ / กระบวนการไม่เสถียร | Week 01–03: CRISP-DM, OODA/DMAIC, EDA, descriptive statistics, visualization, hypotheses, cross-tabs, chi-square, value driver tree | unit/time/KPI baseline, metrics และกลุ่มเทียบ | root-cause candidates, variance/bottleneck/targeted process fixes, financial impact |
| Invoice / Basket แต่ละใบมีหลายสินค้า และถามว่าสินค้าใดซื้อร่วมกัน | Week 04: Association Rule Mining, Apriori, Support, Confidence, Lift | สินค้าเป็น presence per basket, returns/invalids removed or separately analyzed, sparse encoding, parameter ranges | bundling, cross-selling, shelf/stock recommendations, rule stability, A/B, incremental margin |
| Customer-level traits/behavior ไม่มี target label และถามว่ามีกี่ประเภท | Week 05: Segmentation/STP, StandardScaler → K-Means → WCSS/Elbow, Silhouette → centroid profiling, PCA visualization | meaningful distance features, outliers, feature scaling, cluster-size check | personas, tailored campaigns/pricing/service, compare business utility and monitor drift |
| Customer repeat-purchase event log และถามว่าลูกค้าควรลงทุนรักษามากแค่ไหนในอนาคต | Week 06 Enterprise: RFM-T → BG/NBD → Gamma-Gamma (เมื่อสมมติฐานเหมาะ) → discounted CLV | valid purchase events, purchase dates, customer IDs, positive monetary values, customer summary, margin and discount assumptions | expected transactions, expected order value, CLV tier, LTV/CAC allocation and retention/expansion actions |
| Transaction history มี observation และ future target และถามว่าใน 90 วันถัดไปใครจะใช้จ่ายเท่าไร | Week 06 Case Study 2: supervised XGBoost regression เปรียบเทียบ behavioral RFM กับ monthly spend momentum | 9-month observation / 3-month target (ตามตัวอย่าง), customer aggregation, as-of cutoff, strict train/test protocol | predicted 90-day spend, MAE/RMSE/R², whales/sleeping giants/ranking/action; ไม่เท่ากับ discounted lifetime CLV |
| Customer snapshots มี churn labels (หรือกำหนด churn event ได้) | Week 08–09: define churn rule/time t0/horizon → behavioral RFM/trend features → cost-sensitive classification (เช่น Random Forest) → validation threshold/calibration → SHAP → uplift/A/B | timestamp validity, distinct customer snapshots, leakage audit, class balance, train/validation/test | risk ranking, precision/recall/PR-AUC/ROC-AUC, expected intervention value, targeted retention (causal lift ต้องทดสอบ) |
| Social posts/reviews/comments ที่เป็นข้อความ | Week 10: NLP, source provenance and text audit, sentiment classification (VADER baseline or TF-IDF + Logistic Regression), keyphrase extraction, NMF topic discovery, source-supported summaries | raw texts, date/platform/language, label guide or explicitly unsupervised mode, avoid duplicate author/text leakage, preserve negation/emoji | sentiment by period/platform/aspect, recurring themes, complaints, human-reviewed summaries, denominators and response owners |

## C. Method-specific guardrails

**EDA — Week 03.** เมื่อข้อมูล skewed อย่าตีความ mean อย่างเดียว; ใช้ median/IQR ร่วมด้วย แยก variability ออกจาก average; cross-tab เป็น association, chi-square ต้องตรวจ expected counts/สมมติฐานก่อนอ้างนัยสำคัญ; แยก univariate/bivariate/multivariate; เชื่อมถึง KPI และ financial line items.

**Market Basket — Week 04.** Support = P(X∪Y), Confidence(X→Y) = P(Y|X), Lift = P(Y|X)/P(Y). Lift > 1 บ่งชี้ co-occurrence มากกว่ากรณี independence ภายใต้ข้อมูล ไม่ใช่ผลเชิงสาเหตุ. Apriori อาศัย downward closure pruning; ปรับ min_support/min_confidence/max_len เพื่อควบคุม combinatorial explosion และหลีกเลี่ยง rule ที่อิงจำนวน basket น้อย. ประเมิน incremental *margin*, cannibalization, stock and intervention cost ก่อนใช้งาน.

**Segmentation — Week 05.** Fixed demographic/RFM bins เป็น baseline ที่อาจ arbitrary และสูญเสีย interaction; ใช้ StandardScaler ก่อน Euclidean K-Means; เลือก K ด้วย WCSS/Elbow และ Silhouette ประกอบกับความชัดเจน/ขนาดของแต่ละกลุ่ม; PCA 2D เป็น visualization; profiling centroid และออกแบบ STP/A-B campaigns; ไม่ถือว่า cluster ที่ได้มีคุณค่าอัตโนมัติ.

**CLV — Week 06.** RFM-T ของ BG/NBD ใช้ purchase-frequency (repeat events), recency ช่วง first-to-last qualifying purchase, อายุความสัมพันธ์ T (ไม่ใช่ recency-days-since-last-purchase แบบ RFM ดั้งเดิม). BG/NBD ประมาณการซื้อซ้ำ/ยัง active, Gamma-Gamma ประมาณเงินต่อครั้งภายใต้สมมติฐานของโมเดล, แล้วคำนวณ discounted future value และ margin. โปรดระบุข้อจำกัดและความแตกต่างระหว่าง predicted gross spend กับ expected profit CLV. การ clean return/negative quantity ควรเก็บบัญชีผลกระทบแยกไว้เมื่อความเสี่ยงจากคืนสินค้าสำคัญ.

**Churn — Week 08–09.** นิยาม contract/behavioral churn, observation [t0−window,t0), decision t0, label (t0,t0+horizon]. Churn risk P(churn|X) ต่างจาก treatment uplift E[Y(1)−Y(0)|X]; SHAP อธิบาย fitted model ไม่ใช่เหตุ. Class imbalance ต้องดู baseline/precision/recall/PR-AUC และ optimize threshold ด้วยต้นทุน; freeze final decision rule ก่อน test; dataset ที่ไม่มี cutoff timestamps รองรับคำอธิบาย historical association เท่านั้น.

**NLP — Week 10.** แยก classification ที่มี labels จาก topic/keyphrase discovery ที่ไม่มี gold labels; clean text โดยเก็บ raw text และรักษา negation/slang/hashtags/emoji; วัด Macro-F1, per-class recall, confusion matrix สำหรับ sentiment, human interpretation + stability สำหรับ NMF topics; metrics ของ summary/translation ไม่รับรอง factual faithfulness; visualizations ระบุ denominator, period, source และ population.

## D. Checklist ก่อนรับ Task ใหม่

- คำถาม: ธุรกิจต้องการ **อธิบาย / หา pattern / แบ่งกลุ่ม / ทำนายจำนวน / ทำนายความเสี่ยง / วิเคราะห์ข้อความ / ทดสอบผลการกระทำ** อะไร?
- Granularity: 1 row = อะไร? มี ID, timestamp, target labels, action records, revenue, cost, margin หรือไม่?
- Data quality: ที่มาของข้อมูลและ sample, duplicates, missing, invalids, returns, extreme values, category encoding, quantity/price/currency, temporal availability?
- Evaluation: baseline, suitable technical metrics, out-of-sample design, leakage controls, assumptions, uncertainty, and business KPIs?
- Actionability: ใครต้องทำอะไร, ทำไมกับใคร, expected incremental financial value เท่าใด, วัด causal effect และ monitor อย่างไร?

**Scope note:** เอกสารที่ให้ตรวจมี Week 01–06 และ Week 08–10; ไม่มี Week 07. ตารางเป็นการบูรณาการหลักการจากสไลด์ ไม่ใช่การอ้างว่าอาจารย์สั่งให้ใช้ทุกวิธีในทุก Dataset. ค่าพารามิเตอร์ ตัวเลข ROI และ thresholds ต้องประเมินจากโจทย์/ข้อมูลจริงเสมอ.
