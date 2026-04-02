#  數據分析作品集（Data Analysis Portfolio）

本作品集展示完整的資料分析與機器學習建模流程，涵蓋 **統計分析、分類模型、回歸預測、分群分析、關聯規則與時間序列預測**。

專案核心強調：
- 數值型資料建模能力
- 統計方法與機器學習整合
- 模型比較與最佳化
- 預測能力與評估指標（RMSE / MAE / MAPE / Accuracy / AUC）

---

##  專案內容（依資料集分類）

---

###  Iris（多分類問題）

**目標**  
建立多分類模型預測花卉種類（3類）。

**方法與模型**
- KNN（最佳 k 值搜尋）
- Naive Bayes
- Decision Tree（CART）
- SVM
- Logistic Regression（多分類）

**重點技術**
- 模型參數最佳化
- 混淆矩陣與分類準確率評估
- 多模型比較

**成果**
- 比較不同分類模型表現
- 分析線性與非線性模型差異

---

###  Gender_Size（分類 + 分群）

**目標**  
進行性別分類並探索資料潛在分群結構。

**方法與模型**

分類：
- Logistic Regression
- KNN
- SVM

分群：
- K-means
- Hierarchical Clustering
- Gaussian Mixture Model（GMM）

**重點技術**
- 群數選擇（DB index / BIC）
- ROC 曲線與 AUC 評估
- 多數決（Ensemble）

**成果**
- 找出最佳分群數
- 比較監督式與非監督式學習差異

---

###  Pima（醫療二分類）

**目標**  
預測是否患有糖尿病。

**方法與模型**
- Logistic Regression
- KNN
- Decision Tree（CART）
- SVM

**重點技術**
- ROC 曲線與 AUC
- 模型參數最佳化（KNN / SVM / CART）
- 多模型比較

**成果**
- 評估分類模型預測能力
- 找出最佳預測模型

---

###  Titanic（關聯規則 + 分類）

**目標**  
分析乘客生存模式與影響因素。

**方法**
- Apriori 關聯規則
- 分類模型分析

**重點技術**
- Support / Confidence / Lift
- 生存條件規則萃取
- 特徵關係分析

**成果**
- 發現影響生存的重要因素（如性別、艙等）
- 建立可解釋的規則模型

---

###  Wine Quality（Red）（回歸）

**目標**  
預測紅酒品質分數。

**模型**
- Random Forest
- Gradient Boosting
- ANN（類神經網路）

**重點技術**
- 回歸模型比較
- 評估指標（MSE / MAE / MAPE）
- 特徵影響分析

**成果**
- 找出最佳回歸模型
- 提升預測準確度

---

###  Wine Quality（White）（回歸 + ANN最佳化）

**目標**  
透過神經網路提升預測效果。

**方法**
- ANN（nnet）
- 隱藏層神經元調整（size tuning）

**重點技術**
- 超參數調整（size / decay）
- 收斂控制（maxit / rang）
- 模型最佳化

**成果**
- 找出最佳神經網路架構
- 改善預測表現

---

###  Breast Cancer（分類 + 交叉驗證）

**目標**  
分類腫瘤為良性或惡性。

**模型**
- Logistic Regression
- Naive Bayes
- Decision Tree
- ANN
- SVM

**重點技術**
- K-fold Cross Validation
- 平均表現比較（AVG）
- 模型穩定性分析

**成果**
- 比較模型穩定性與準確率
- 找出最佳分類模型

---

###  Bank（集成學習）

**目標**  
預測客戶是否訂閱產品。

**模型**
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- XGBoost

**重點技術**
- Ensemble Learning 比較
- ROC / AUC 評估
- 特徵重要度分析

**成果**
- 比較 Boosting 與 Bagging 表現
- 找出最佳集成模型

---

###  時間序列（預測分析）

（Electronics / Revenue / Electricity / Material）

**目標**  
預測未來時間序列趨勢。

**模型**
- ARIMA / SARIMA
- Holt-Winters
- VAR
- Dynamic ARIMA

**重點技術**
- AIC 選擇最佳參數
- 季節性建模（P,D,Q,s）
- 多變量時間序列分析

**成果**
- 比較不同預測模型效果
- 分析外部變數影響

---

##  技能總結

- 統計建模與資料分析
- 機器學習（分類 / 回歸）
- 分群分析與關聯規則
- 時間序列預測
- 模型評估與最佳化
- R / Python 資料分析

---

##  作品集特色

- 完整流程：資料處理 → 建模 → 評估
- 強調數值建模能力（非僅視覺化）
- 多模型比較與最佳化
- 可應用於實務預測問題
