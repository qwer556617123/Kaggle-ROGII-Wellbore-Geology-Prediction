import pandas as pd
from pathlib import Path

for wid in ['000d7d20', '00bbac68', '00e12e8b']:
    test = pd.read_csv(f'test/{wid}__horizontal_well.csv')
    train = pd.read_csv(f'train/{wid}__horizontal_well.csv')
    
    test_mask = test['TVT_input'].isna() | (test['TVT_input'].astype(str).str.strip() == '')
    ps = int(test_mask.idxmax()) if test_mask.any() else len(test)
    
    print(f'{wid}:')
    print(f'  test rows={len(test)}, train rows={len(train)}, ps={ps}')
    
    if len(train) != len(test):
        print(f'  !!! ROW COUNT MISMATCH: train={len(train)}, test={len(test)}')
    
    # MD alignment check
    min_len = min(len(train), len(test))
    md_diff = (train['MD'][:min_len] - test['MD'][:min_len]).abs().max()
    print(f'  MD max diff: {md_diff:.6f}')
    
    # Pre-PS TVT match
    pre_match = (train['TVT'][:ps] - test['TVT_input'][:ps]).abs().max()
    print(f'  Pre-PS TVT match (max diff): {pre_match:.6f}')
    
    print(f'  TVT at ps-1: train={train["TVT"].iloc[ps-1]:.4f}, test_input={test["TVT_input"].iloc[ps-1]:.4f}')
    
    # Compare TVT_input from test with TVT from train for post-PS rows
    # (train has full TVT, test has NaN for post-PS TVT_input)
    post_train_tvt = train['TVT'].values[ps:ps+5]
    print(f'  First 5 post-PS TVT in train: {post_train_tvt}')
    print()
