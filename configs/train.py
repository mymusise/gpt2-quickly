path = '/data/wiki_zh'
model_path = path + '/models/'

data = {
    'path': path + '/train/',
}
data = {
    **data,
    'raw': data['path'] + 'raw.txt',
    'raw_cut': data['path'] + 'raw.cut.txt',
    'vocab': data['path'] + 'vocab.txt',
    'pickle': data['path'] + 'data.pickle',
}

model = {
    'max_length': 1024,
    'n_positions': 1024,
    'n_ctx': 1024,
    'n_embd': 1024,
    'n_layer': 24,
    'n_head': 16,
    'batch_size': 6
}


data = type('data', (), data)
model = type('model', (), model)
