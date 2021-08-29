import os
import requests


def main(url, dataset):
    # download
    print(f'Download {url} ...')
    response = requests.get(url)
    with open(f'datasets/example/{dataset}.zip', 'wb') as f:
        f.write(response.content)

    # unzip
    import zipfile
    with zipfile.ZipFile(f'datasets/example/{dataset}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'datasets/example/{dataset}')


if __name__ == '__main__':
    """This script will download and prepare the example data:
        1. BSDS100 for training
        2. Set5 for testing
    """
    os.makedirs('datasets/example', exist_ok=True)

    urls = [
        'https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/BSDS100.zip',
        'https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/Set5.zip'
    ]
    datasets = ['BSDS100', 'Set5']
    for url, dataset in zip(urls, datasets):
        main(url, dataset)
