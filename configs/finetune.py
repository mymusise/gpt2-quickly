path = '/data/novels'
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
    'batch_size': 2,
}


data = type('data', (), data)
model = type('model', (), model)
