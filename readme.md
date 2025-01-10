# 基於設定目標之強化學習參數控制

## 專案簡介
實務上，使用者往往會透過一些方法，讓
## 專案動機

## 專案內容

### 實際資料應用
連續攪拌罐式反應器(https://en.wikipedia.org/wiki/Continuous_stirred-tank_reactor)
## 衍生應用

PID 控制器存在以下限制：
1. 參數敏感性：控制效果高度依賴於比例增益（$K_p$）、積分增益（$K_i$）和微分增益（$K_d$）的設定，參數調整困難且需針對不同系統進行調適。
2. 適用範圍有限：主要適用於基本線性、且動態特性不隨時間變化的系統。對於高度非線性或具有時變特性的系統，效果可能不佳。
3. 多變數系統限制：PID 控制器主要應用於單一輸入單一輸出（SISO）系統。在多輸入（MISO）或多輸入多輸出（MIMO）系統中，變數間的相互干擾可能導致控制效果下降，難以實現全局最佳化。

相比之下，使用強化學習方法具有以下優勢：
1.	降低參數依賴性：無需像 PID 控制器那樣高度依賴參數設定，能自動學習最佳控制策略。
2.	支持非線性系統：適用於非線性或具有時變特性的系統，能有效處理複雜的動態行為。
3.	無多變數限制：在多輸入多輸出（MIMO）系統中表現良好，能減少變數間的相互干擾，實現全局最佳化。