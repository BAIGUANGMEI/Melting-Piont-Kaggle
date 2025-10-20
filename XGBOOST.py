import pandas as pd
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 读取训练集数据
data = pd.read_csv("train.csv")

# 计算所有描述符
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({name: None for name, _ in Descriptors._descList})
    return pd.Series({name: func(mol) for name, func in Descriptors._descList})

# 计算分子描述符并合并（这里采用 compute_all_descriptors 也可以用 compute_descriptors）
rdkit_features = data['SMILES'].apply(compute_all_descriptors)
data = pd.concat([data, rdkit_features], axis=1)
data = data.fillna(0)

# 增加分子特征交互
data["MolWt_NumHDonors"] = data["MolWt"] * data["NumHDonors"]
data["MolWt_NumHAcceptors"] = data["MolWt"] * data["NumHAcceptors"]
data["TPSA_MolWt_Ratio"] = data["TPSA"] / (data["MolWt"] + 1e-6)

data["NumDonors_Acceptors_Ratio"] = data["NumHDonors"] / (data["NumHAcceptors"] + 1e-6)
data["MolWt_SumH"] = data["MolWt"] * (data["NumHDonors"] + data["NumHAcceptors"])
data["TPSA_MolWt_SumRatio"] = data["TPSA"] / (data["MolWt_SumH"] + 1e-6)
# 计算分子指纹
# Morgan fingerprint（1024 位，半径2）
def compute_morgan_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*nBits
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# MACCS keys fingerprint（一般为167位）
def compute_maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*167
    fp = MACCSkeys.GenMACCSKeys(mol)
    nBits = fp.GetNumBits()
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()


# 计算 Morgan fingerprint
morgan_fps = data['SMILES'].apply(lambda s: compute_morgan_fp(s, radius=2, nBits=1024))
morgan_df = pd.DataFrame(morgan_fps.tolist(), columns=[f'Morgan_{i}' for i in range(1024)])

# 计算 MACCS keys fingerprint
maccs_fps = data['SMILES'].apply(lambda s: compute_maccs_fp(s))
maccs_df = pd.DataFrame(maccs_fps.tolist(), columns=[f'MACCS_{i}' for i in range(167)])

# 合并指纹到 data
data = pd.concat([data, morgan_df, maccs_df], axis=1)

# -------------------- 特征准备 --------------------
# group 特征：以 "Group" 开头的列
group_columns = [col for col in data.columns if col.startswith("Group")]

# 分子描述符（这里使用基本描述符和交互特征）
descriptor_columns = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
new_feature_columns = ["MolWt_NumHDonors", "MolWt_NumHAcceptors", "TPSA_MolWt_Ratio", "NumDonors_Acceptors_Ratio", "MolWt_SumH", "TPSA_MolWt_SumRatio"]
molecular_feature_columns = descriptor_columns + new_feature_columns

# 指纹特征：Morgan 和 MACCS
morgan_columns = [f'Morgan_{i}' for i in range(1024)]
maccs_columns = [f'MACCS_{i}' for i in range(167)]

# 最终特征集合：group 特征 + 分子描述符 + 指纹特征
final_features = group_columns + molecular_feature_columns + morgan_columns + maccs_columns

X = data[final_features]
y = data["Tm"]

# 划分训练集、测试集（保留 DataFrame 结构，便于后续处理）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 数据归一化（使用 StandardScaler 实现 Zscore 标准化）
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                             columns=X_test.columns, index=X_test.index)

# 定义 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------- 调优 XGBoost --------------------
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500) 
    max_depth = trial.suggest_int('max_depth', 3, 12) 
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0) 
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)  
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)  
    gamma = trial.suggest_float('gamma', 0, 5) 
    
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }
    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=kf,
                            scoring='neg_mean_squared_error', n_jobs=-1)
    return -score.mean()

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)
print("XGBoost 最优参数:", study_xgb.best_params)

# 使用调优后的参数训练 XGBoost 模型
best_xgb = XGBRegressor(**study_xgb.best_params,
                        objective='reg:squarederror',
                        random_state=42,
                        n_jobs=-1)
best_xgb.fit(X_train_scaled, y_train)
y_pred = best_xgb.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(test_mse)
print("XGBoost 测试集 MSE:", test_mse)
print("XGBoost 测试集 RMSE:", rmse)

# 读取 test.csv 进行预测
test_data = pd.read_csv("test.csv")

# 对测试集进行相同的归一化处理及最终特征筛选
rdkit_features_test = test_data['SMILES'].apply(compute_all_descriptors)
test_data = pd.concat([test_data, rdkit_features_test], axis=1)
test_data = test_data.fillna(0)

# 增加交互特征和次数变换
test_data["MolWt_NumHDonors"] = test_data["MolWt"] * test_data["NumHDonors"]
test_data["MolWt_NumHAcceptors"] = test_data["MolWt"] * test_data["NumHAcceptors"]
test_data["TPSA_MolWt_Ratio"] = test_data["TPSA"] / (test_data["MolWt"] + 1e-6)

test_data["NumDonors_Acceptors_Ratio"] = test_data["NumHDonors"] / (test_data["NumHAcceptors"] + 1e-6)
test_data["MolWt_SumH"] = test_data["MolWt"] * (test_data["NumHDonors"] + test_data["NumHAcceptors"])
test_data["TPSA_MolWt_SumRatio"] = test_data["TPSA"] / (test_data["MolWt_SumH"] + 1e-6)


morgan_fps_test = test_data['SMILES'].apply(lambda s: compute_morgan_fp(s, radius=2, nBits=1024))
morgan_df_test = pd.DataFrame(morgan_fps_test.tolist(), columns=[f'Morgan_{i}' for i in range(1024)])
maccs_fps_test = test_data['SMILES'].apply(lambda s: compute_maccs_fp(s))
maccs_df_test = pd.DataFrame(maccs_fps_test.tolist(), columns=[f'MACCS_{i}' for i in range(167)])
test_data = pd.concat([test_data, morgan_df_test, maccs_df_test], axis=1)

X_final_test = test_data[final_features]
X_final_test_scaled = pd.DataFrame(scaler.transform(X_final_test), columns=X_final_test.columns, index=X_final_test.index)

# 预测并保存结果
y_pred_test = best_xgb.predict(X_final_test_scaled)
submission = pd.DataFrame({"id": test_data["id"], "Tm": y_pred_test})
submission.to_csv("submission.csv", index=False)