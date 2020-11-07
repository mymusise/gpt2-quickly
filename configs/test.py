path = './dataset'
model_path = path + '/models/'

data = {
    'path': path + '/test/',
}
data = {
    **data,
    'raw': data['path'] + 'raw.txt',
    'vocab': data['path'] + 'vocab.txt',
    'pickle': data['path'] + 'data.pickle',
}

model = {
    'max_length': 64,
    'n_positions': 512,
    'n_ctx': 512,
    'n_embd': 768,
    'n_layer': 8,
    'n_head': 8,
    'batch_size': 6
}


data = type('data', (), data)
model = type('model', (), model)
