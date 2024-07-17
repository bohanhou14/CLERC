import os
import pickle
import json


def cache_run(func):
    def inner(*args, **kwargs):
        cache_path = kwargs.pop('cache_path', None)
        force_update = kwargs.pop('force_update', False)
        if cache_path is not None and os.path.exists(cache_path) and not force_update:
            if cache_path.endswith('.pkl'):
                return pickle.load(open(cache_path, 'rb'))
            elif cache_path.endswith('.npz'):
                from scipy.sparse import load_npz
                return load_npz(cache_path)
            elif cache_path.endswith('.jsonl'):
                return list(map(json.loads, open(cache_path)))
            elif cache_path.endswith('.json'):
                return json.load(open(cache_path))
            else:
                return open(cache_path).read()
        ret = func(*args, **kwargs)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if cache_path.endswith('.pkl'):
                pickle.dump(ret, open(cache_path, 'wb'))
            elif cache_path.endswith('.npz'):
                from scipy.sparse import save_npz
                save_npz(open(cache_path, 'wb'), ret)
            elif cache_path.endswith('.jsonl'):
                open(cache_path, 'w').write('\n'.join(map(json.dumps, ret)))
            elif cache_path.endswith('.json'):
                json.dump(ret, open(cache_path, 'w'))
            else:
                open(cache_path, 'w').write(ret)
        return ret
    return inner
