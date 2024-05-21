import json

with open('config.json') as config_file:
    config = json.load(config_file)

with open('.env', 'w') as env_file:
    for key, value in config.items():
        env_file.write(f'{key}={value}\n')
