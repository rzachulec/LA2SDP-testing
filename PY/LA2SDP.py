# packages -------------------------------------------------------------

# faster excel imports
from python_calamine.pandas import pandas_monkeypatch
# get terminal width with shutil
import shutil
import sys
import os
import time
import matplotlib.pyplot as plt
# self explanatory
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
            matthews_corrcoef, accuracy_score, recall_score,
            precision_score, fbeta_score, roc_auc_score, confusion_matrix
        )
from sklearn.model_selection import StratifiedShuffleSplit
import dalex as dx

# variables ------------------------------------------------------------

# set amound of output, from 1 to 2
verbosity = 1

dataset_path = '../'

#keep this in a list, thus can be adjusted:)
dataset_names = ['QCdata_1', 'QCdata_2', 'QCdata_3', 'QCdata_4', 'QCdata_5', 'QCdata_6']

categorical_cols = ['AUTOMATION.LEVEL.FINAL', 'TEST.AUTOMATION.LEVEL', 'TEST.OBJECT', 'PROGRAM.PHASE', 'RELEASE', 'TEST.ENTITY', 'network', 'project', 'location_raw', 'suffix']

model_params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'MCC',
    'verbose': 0,
    # 'auto_class_weights': 'Balanced',
    'random_seed': 42
}
    
windows = [
    '2_3', '3_4', '4_5', '5_6',
    '1_3', '2_4', '3_5', '4_6',
    '1_4', '2_5', '3_6',
    '1_5', '2_6',
    '1_6',
    
    # '1_2', '1_3', '1_4', '1_5', '1_6', 
    # '2_3', '2_4', '2_5', '2_6', 
    # '3_4', '3_5', '3_6', 
    # '4_5', '4_6', 
    # '5_6' 
]


# use a dictionary to hold data to enable iterating over filenames in all functions (leave EMPTY for now!)
data = {}
tasks = {}
custom_resamplings = {}
all_results = []

term_width = shutil.get_terminal_size((100, 20)).columns
bar_width = term_width - 10

# contents of main() ---------------------------------------------------

# check gpu backend
def check_backend():
    # check if apple silicon GPU backend is recognised
    # expected ouput: 'tensor([1.], device='mps:0')'
    if (os.uname().sysname == 'Darwin'):
        print('Checking GPU backed:')
        if torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            x = torch.ones(1, device=mps_device)
            print (x)
            print('MPS device found, GPU backend for Apple Silicon enabled.\n')
        else:
            print ('MPS device not found.')
    
# load datasets
def load_datasets():
    if verbosity > 0:
        print('-' * 40)
        print('Loading datasets:')
    for dataset in dataset_names:
        if verbosity > 0:
            print(f'Loading {dataset}...')
        file_path = os.path.join(dataset_path, f'{dataset}.xlsx')
        data[dataset] = pd.read_excel(file_path, engine='calamine')
        if verbosity > 1:
            print(data[dataset].head(), '\n')
    if verbosity > 0:
        print('-' * 40)
    return data

# preprocess datasets
def preprocess_datasets():
    print('\n')
    print('-' * 40)
    print('Preprocessing')
    for dataset in dataset_names:
        df = data[dataset]
            
        # extract info from org col by parsing with helper function        
        parsed_df = parse_organization_batch(df['ORGANIZATION'])
        df = pd.concat([df, parsed_df], axis=1)
        
        if 'EXECUTION.DATE' in df.columns:
            df['EXECUTION.DATE'] = pd.to_datetime(df['EXECUTION.DATE'], errors='coerce')
            df['week'] = df['EXECUTION.DATE'].dt.isocalendar().week
            df['wday'] = df['EXECUTION.DATE'].dt.dayofweek + 1  # mon=1, sun=7
                
        # drop columns from original dropped list:
        cols_to_drop = [
            'EXECUTION.DATE',
            'FAULT.REPORT.ID',
            'FAULT.CREATION.DATE',
            'DOMAIN',
            'PROJECT',
            'FAULT.REPORT.NB',
            'AUTOMATION.LEVEL',
            'DETAILED.AUTOMATION.LEVEL',
            'TEST.RUN.ID',
            'ORGANIZATION'
        ]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # drop columns where at least 20% of values is not NaN
        cnt = len(df)
        df = df.dropna(axis=1, thresh=cnt*0.2)
        
        df[categorical_cols] = df[categorical_cols].fillna('missing')
        df['TEST.STATUS'] = df['TEST.STATUS'].replace({'Passed': 0, 'Failed': 1}).astype(int)

        # missing_by_class = df.groupby('TEST.STATUS').apply(lambda g: g.isna().sum())
        # print('missing:', missing_by_class.T)
        
        # drop rows witn NaN in any col
        df = df.dropna(axis=0, how='any')
        
        
        data[dataset] = df
        if verbosity > 1:
            print(data[dataset].count(), '\n')
            # print(data[dataset].head(), '\n')
            print(data[dataset], '\n')
        out = os.path.join(dataset_path, 'out.csv')
        data[dataset].to_csv(out)
    print('-' * 40)

# prepare dict of tasks
def prepare_tasks():
    if verbosity > 0:
        print()
        print('-' * 40)
        print('Preparing tasks...')
        
    for window in windows:
        task_name = f'task_{window}'        
        
        start, end = map(int, window.split('_'))
        
        if verbosity > 1:
            print(f'\nTask_{window}')
            
        # build task data from QC[start] to QC[end]
        task_df = build_window_data(data, start, end)
        # define index ranges
        train_idx, test_idx = time_based_partition(len=len(task_df), ratio=(1-len(data[f'QCdata_{end}'])/len(task_df)))
            
        task = ClassificationTask(task_name, task_df, 'TEST.STATUS')        
        # can probably be one dict but then train_idx doesnt hold value for some reason!!!
        tasks[task_name] = {
            'name': task_name,
            'task': task,
        }
        custom_resamplings[task_name] = {
            'train_idx': train_idx,
            'test_idx': test_idx
        }
            
    if verbosity > 0:
        print(tasks.keys())
        print('-' * 40)

# train model on prepared tasks
def train_tasks():
    if verbosity > 0:
        print()
        print('-' * 40)
        print('Training model...')

    for idx, task_name in enumerate(tasks.keys()):
        task_info = tasks[task_name]
        split = custom_resamplings[task_name]

        if verbosity > 1:
            print('\n')
            print(f'Task Name: {task_info['name']}')
            start, end = map(int, task_name[-3:].split('_'))
            print(f'Train window: QCdata_{start}_{end-1}')
            print(f'Test window: QCdata_{end}')    

        train_pool, test_pool, y_test = get_catboost_pool(task_info['task'], split['train_idx'], split['test_idx'])

        start_train = time.time()
        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=test_pool)
        train_time = time.time() - start_train

        start_pred = time.time()
        preds = model.predict(test_pool)
        pred_time = time.time() - start_pred

        metrics(y_test, preds, task_info['name'], model, test_pool, train_time=train_time, pred_time=pred_time)

        if verbosity > 0:
            print_progress_bar(idx + 1, len(windows), width=bar_width)
            print('\n')

    if verbosity > 0:
        print('-' * 40)


def feature_imp():
    if verbosity > 0:
        print()
        print('-' * 40)
        print('Running feature importance...')
    task_df = build_window_data(data, 1, len(data))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in sss.split(task_df, task_df['TEST.STATUS']):
        train_df = task_df.iloc[train_index]
        test_df = task_df.iloc[test_index]

    X_train = train_df.drop(columns='TEST.STATUS')
    y_train = train_df['TEST.STATUS']
    X_test = test_df.drop(columns='TEST.STATUS')
    y_test = test_df['TEST.STATUS']

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)
        
    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=test_pool)

    print(y_test.head())
    # y_test = y_test.replace({'Passed': 1, 'Failed': 0}).astype(int)
    
    explainer = dx.Explainer(model, X_test, y_test, model_type='classification', label='CatBoost')
    
    fi = explainer.model_parts(loss_function=one_minus_mcc, type='permutational')
    fig = fi.plot(show=True)
    if verbosity > 0:
        print('-' * 40)


# helper functions -------------------------------------------------------


def one_minus_mcc(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return 1 - matthews_corrcoef(y_true, y_pred_labels)

# used in preprocess_datasets()
# parse 'ORGANIZATION' col into separate cols
def parse_organization_batch(series):
    parsed_rows = []

    for org in series:
        org = str(org).strip()

        parsed = {
            'network': None,
            'project': None,
            # 'country': None,
            # 'city': None,
            'location_raw': None,
            'suffix': None
        }
        
        parts = org.split('_')
        parsed.update({
            'network': parts[0] if len(parts) > 0 else None,
            'project': parts[1] if len(parts) > 1 else None,
            'location_raw': parts[2] if len(parts) > 2 else None,
            'suffix': parts[-1] if len(parts) > 3 else None
        })

        parsed_rows.append(parsed)

    return pd.DataFrame(parsed_rows)

# used in prepare_tasks()
# create task_df form imported datasets
def build_window_data(data, start, end):
    frames = [data[f'QCdata_{str(i)}'] for i in range(start, end)]
    if verbosity > 1:
        print(f'\nFrames: {frames.index}\n')
    return pd.concat(frames, ignore_index=True)

# used in prepare_tasks()
# helper class to align the syntax of  a task with the R code
class ClassificationTask:
    def __init__(self, name, df, target):
        self.name = name
        self.df = df
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.n = len(df)
        
# used in prepare_tasks()
# used to determine indexes of train and test parts of task_df
def time_based_partition(len, ratio):
    '''returns indexe to be used wihthin task_df in order to achieve a chronological
    split between training and test data.

    Args:
        len (int): length of task_df
        test_size (_type_): fraction of task_df which will be used for testing

    Returns:
        train_idx (list(int)): list of training indexes
        test_idx (list(int)): list of testing indexes
    '''    
    ratio = 0.67
    n = len
    n_test = int((1 - ratio) * n)
    n_train = n - n_test
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n))
    return train_idx, test_idx

# used in train_tasks()
# wrapper for catboost Pool funciton to generate model input sets
def get_catboost_pool(task, train_idx, test_idx):
    X = task.X
    y = task.y
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

    return train_pool, test_pool, y_test

# used in train_tasks()
# function to aggregate, display and save model metrics
def metrics(y_test, preds, task_name, model, test_pool, train_time=None, pred_time=None):
    
    mcc = matthews_corrcoef(y_test, preds)
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    fbeta = fbeta_score(y_test, preds, beta=1)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    try:
        proba = model.predict_proba(test_pool)
        if len(model.classes_) == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class='ovr')
    except Exception:
        auc = None

    if verbosity > 1:
        print(f'\n{task_name} results:')
        print(f'MCC: {mcc:.4f}')
        print(f'Accuracy: {acc:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F-beta: {fbeta:.4f}')
        print(f'AUC: {auc:.4f}' if auc is not None else 'AUC: N/A')
        print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
        print(f'Train time: {train_time:.3f}s, Predict time: {pred_time:.3f}s')

    results = {
        'task': task_name,
        'MCC': mcc,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'F-beta': fbeta,
        'AUC': auc if auc is not None else float('nan'),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Train Time (s)': train_time if train_time is not None else float('nan'),
        'Predict Time (s)': pred_time if pred_time is not None else float('nan')
    }

    all_results.append(results)

    results_df = pd.DataFrame(all_results)

    numeric_cols = results_df.select_dtypes(include='number').columns
    avg_row = results_df[numeric_cols].mean(numeric_only=True)
    avg_row['task'] = 'AVERAGE'

    results_df_with_avg = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    results_df_with_avg = results_df_with_avg.round(3)

    results_df_with_avg.to_csv(os.path.join(dataset_path, 'results-4.csv'), index=False)



def print_progress_bar(current, total, width=80):
    progress = int((current / total) * width)
    bar = '#' * progress + '-' * (width - progress)
    percent = (current / total) * 100
    sys.stdout.write(f'\r[{bar}] {percent:6.2f}%')
    sys.stdout.flush()


# ------------------------------------------------------------------------

def main():
    check_backend()
    load_datasets()
    preprocess_datasets()
    prepare_tasks()
    train_tasks()
    feature_imp()
        
if __name__ == '__main__':
    main()