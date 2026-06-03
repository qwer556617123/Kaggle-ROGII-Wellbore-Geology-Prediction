import pickle, numpy as np, pandas as pd

with open('models/lgbm_v8.pkl', 'rb') as f:
    model = pickle.load(f)

print('Type:', type(model))
names = model.feature_name()
imps  = model.feature_importance(importance_type='gain')
fi = sorted(zip(imps, names), reverse=True)
print('\nTop 20 features (gain):')
for imp, name in fi[:20]:
    print(f'  {imp:>10.1f}  {name}')
print(f'\nTotal: {len(names)}')
