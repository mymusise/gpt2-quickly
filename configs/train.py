path = '/data/wiki_zh'
model_path = path + '/models/'

data = {
    'path': path + '/train/',
}
data['raw'] = data['path'] + 'raw.txt'
data['vocab'] = data['path'] + 'vocab.txt'
data['pickle'] = data['path'] + 'data.pickle'


model = {
    'max_length': 512,
    'n_positions': 512,
    'n_ctx': 512,
    'n_embd': 768,
    'n_layer': 24,
    'n_head': 24,
    'batch_size': 2
}


data = type('data', (), data)
model = type('model', (), model)
