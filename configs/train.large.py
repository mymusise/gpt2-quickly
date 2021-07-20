path = '/content/drive/MyDrive/100word'
model_path = path + '/models/'

data = {
    'path': path + '/train/',
}
data = {
    **data,
    'raw': data['path'] + 'raw.txt',
    'vocab': data['path'] + 'vocab.txt',
    'pickle': data['path'] + 'data.pickle',
}

model = {
    'max_length': 1024,
    'n_positions': 1024,
    'n_ctx': 1024,
    'n_embd': 1280,
    'n_layer': 36,
    'n_head': 20,
    'batch_size': 2
}


data = type('data', (), data)
model = type('model', (), model)
