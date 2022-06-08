import requests
from zipfile import ZipFile
from pathlib import Path

DATASETS = [
    'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/pretrain/pre-training_computers_only_new_15.json.gz',
    'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2_nonnorm/offers_corpus_english_v2_non_norm.json.gz'
]


def download_datasets():
    for link in DATASETS:

        '''iterate through all links in DATASETS 
        and download them one by one'''

        # obtain filename by splitting url and getting
        # last string
        file_name = link.split('/')[-1]

        print("Downloading file:%s" % file_name)

        # create response object
        r = requests.get(link, stream=True)

        # download started
        with open(f'../../data/raw/wdc-lspc/{file_name}', 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print("%s downloaded!\n" % file_name)

    print("All files downloaded!")
    return

if __name__ == "__main__":
    Path('../../data/raw/wdc-lspc/').mkdir(parents=True, exist_ok=True)
    download_datasets()
