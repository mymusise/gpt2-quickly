path = './dataset'
model_path = path + '/models/'

data = {
    'path': path + '/test/',
}
data = {
    **data,
    'raw': data['path'] + 'raw.txt',
    'raw_cut': data['path'] + 'raw.cut.txt',
    'vocab': data['path'] + 'vocab.txt',
    'pickle': data['path'] + 'data.pickle',
}

model = {
    'max_length': 64,
    'n_positions': 512,
    'n_ctx': 512,
    'n_embd': 768,
    'n_layer': 4,
    'n_head': 4,
    'batch_size': 8
}


data = type('data', (), data)
model = type('model', (), model)
