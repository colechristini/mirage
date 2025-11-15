import json, os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV,  KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from scipy.stats import randint, uniform
import lightgbm as lgb
import optuna

def load_json_file(file_path):
   with open(file_path, 'r') as file:
       return json.load(file)
   
def process_by_guid(data):
  out = {}
  for item in data:
    for output in item['output_tensors']:
      guid = output['guid']
      out[guid] = item['op_type']
  return out

def generate_nsm(graph, by_guid, op_types):
  idxs = {op_type: i for i, op_type in enumerate(op_types)}
  n_ops = len(op_types)
  nsm = np.zeros((n_ops, n_ops), dtype=np.int8)
  for op in graph: 
    op_type = op["op_type"]
    if op_type in ["kn_input_op", "kn_output_op"]:
      continue
    idx = idxs[op_type]
    for input_tensor in op["input_tensors"]:
      input_op_type = by_guid[input_tensor["guid"]]
      if input_op_type in ["kn_input_op", "kn_output_op"]:
        continue
      input_idx = idxs[input_op_type]
      nsm[input_idx, idx] += 1
  return nsm.ravel()


def get_all_op_types(data_list):
  op_types = set()
  for item in data_list:
    for op in item:
      if op["op_type"] not in ["kn_input_op", "kn_output_op"]:
        op_types.add(op["op_type"])
  return list(op_types)

def compute_flops(op_type, input_tensors):
  if op_type == 'kn_matmul_op':
    M = input_tensors[0][-2]
    K = input_tensors[0][-1]
    N = input_tensors[1][-1]
    return 2 * M * N * K
  elif op_type in ['kn_add_op', 
                    'kn_div_op', 
                    'kn_exp_op', 
                    'kn_mul_op', 
                    'kn_pow_op', 
                    'kn_sqrt_op', 
                    'kn_reduction_2_op']:
    num_elements = 1
    for dim in input_tensors[0]:
      num_elements *= dim
    return num_elements
  elif op_type in ['kn_input_op', 'kn_output_op']:
    return 0
  else:
    raise ValueError(f"Unknown op type: {op_type}")
  
def compute_total_flops(file_data):
    total_flops = 0
    for node in file_data:
      op_type = node['op_type']
      in_tensors = []
      for t in node['input_tensors']:
        in_tensors.append(t['dim'][:-1])
      flops = compute_flops(op_type, in_tensors)
      print(op_type, flops)
      total_flops += compute_flops(op_type, in_tensors)
      
    return total_flops

def generate_features_and_labels(data, label_file):
  op_types = get_all_op_types(list(data.values()))
  for file_id, file_data in data.items():
    data[file_id] = (file_data, process_by_guid(file_data))
  features = []
  labels = []
  tf_arr = []
  for file_id, (file_data, by_guid) in data.items():
    nsm = generate_nsm(file_data, by_guid, op_types)
    print(type(nsm))
    total_flops = compute_total_flops(file_data)
    tf_arr.append(total_flops)
    if file_id in label_file:
      features.append(nsm)
      labels.append(label_file[file_id])
      print(f"Processed file {file_id}")
    else:
      print(f"Warning: No performance data found for file {file_id}")
  tf_arr = np.array(tf_arr)
  features = np.array(features)
  features = np.concatenate((features, tf_arr.reshape(len(tf_arr), 1)), axis=1)
  labels = np.array(labels)
  return features, labels

def pairwise_ranking_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]
    mask = diff_true != 0
    disagreements = np.sum((diff_true * diff_pred < 0)[mask])
    total_pairs = np.sum(mask)
    return disagreements / total_pairs if total_pairs > 0 else 0.0

def train_rf(features: np.ndarray, labels: np.ndarray, n_splits: int = 5, n_iter: int = 20, random_state: int = 42):
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    param_dist = {
        'n_estimators': randint(50, 150),
        'max_depth': randint(2, 15),
        'min_samples_split': randint(2, 6),
        'min_samples_leaf': randint(1, 4),
        'max_features': uniform(0.5, 0.4)
    }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=rmse_scorer,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    search.fit(features, labels)
    best_params = search.best_params_
    best_rmse = -search.best_score_

    final_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
    final_model.fit(features, labels)

    cv_scores_r2 = cross_val_score(final_model, features, labels, cv=cv, scoring='r2', n_jobs=-1)
    cv_scores_rmse = -cross_val_score(final_model, features, labels, cv=cv, scoring=rmse_scorer, n_jobs=-1)
    cv_scores_mae = -cross_val_score(final_model, features, labels, cv=cv, scoring=mae_scorer, n_jobs=-1)

    ranking_losses = []
    for train_idx, test_idx in cv.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        ranking_losses.append(pairwise_ranking_loss(y_test, y_pred))
    mean_rank_loss = np.mean(ranking_losses)
    std_rank_loss = np.std(ranking_losses)

    results = {
        "best_params": best_params,
        "mean_cv_rmse": best_rmse,
        "mean_r2": cv_scores_r2.mean(),
        "std_r2": cv_scores_r2.std(),
        "mean_rmse": cv_scores_rmse.mean(),
        "std_rmse": cv_scores_rmse.std(),
        "mean_mae": cv_scores_mae.mean(),
        "std_mae": cv_scores_mae.std(),
        "mean_rank_loss": mean_rank_loss,
        "std_rank_loss": std_rank_loss
    }

    print("\n=== Random Forest Training Summary ===")
    print("Best hyperparameters:", best_params)
    print(f"Cross-validated RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"Cross-validated MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
    print(f"Cross-validated R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"Pairwise Ranking Accuracy: {1 - results['mean_rank_loss']:.4f} ± {results['std_rank_loss']:.4f}")

    return final_model, results


if __name__ == '__main__':
  files = os.listdir('data')
  files = [os.path.join('data', f) for f in files if 'original' in f]
  data = {}
  for file in files:
    file_id = file.split("_")[1].split(".")[0]
    data[file_id] = load_json_file(file)
  label_file = load_json_file('performance.json')
  features, labels = generate_features_and_labels(data, label_file)
  print("Number of files processed:", len(files))
  print("Features shape:", features.shape)
  train_rf(features, labels)