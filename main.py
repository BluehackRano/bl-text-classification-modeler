from __future__ import print_function

import os
from util import s3
import redis
import fasttext
from bluelens_log import Logging
from bluelens_spawning_pool import spawning_pool
from stylelens_dataset.texts import Texts
from stylelens_product.products import Products

from random import shuffle

SPAWN_ID = os.environ['SPAWN_ID']
REDIS_SERVER = os.environ['REDIS_SERVER']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']
RELEASE_MODE = os.environ['RELEASE_MODE']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')

options = {
  'REDIS_SERVER': REDIS_SERVER,
  'REDIS_PASSWORD': REDIS_PASSWORD
}
log = Logging(options, tag='bl-text-classification-modeler')
rconn = redis.StrictRedis(REDIS_SERVER, decode_responses=True, port=6379, password=REDIS_PASSWORD)

storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

text_api = Texts()
product_api = Products()

TEXT_CLASSIFICATION_MODEL = 'text_classification_model'
DATASET_LABEL_PREFIX = '__label__'
generated_datasets = []

def delete_pod():
  log.info('exit: ' + SPAWN_ID)

  data = {}
  data['namespace'] = RELEASE_MODE
  data['key'] = 'SPAWN_ID'
  data['value'] = SPAWN_ID
  spawn = spawning_pool.SpawningPool()
  spawn.setServerUrl(REDIS_SERVER)
  spawn.setServerPassword(REDIS_PASSWORD)
  spawn.delete(data)

def save_model_to_storage():
  #Todo: Need to implement by rano@bljuehack.net
  log.info('save_model_to_storage')

def retrieve_keywords(class_code):
  offset = 0
  limit = 100
  while True:
    keywords = text_api.get_texts(class_code, offset=offset, limit=limit)
    retrieve_products(class_code, keywords)

    if limit > len(keywords):
      break
    else:
      offset = offset + limit

def convert_dataset_as_fasttext(class_code, datasets):
  log.info('convert_dataset_as_fasttext')

  for dataset in datasets:
    count = len(dataset)
    if count > 5:
      count = 5
    for n in range(count):
      shuffle(dataset)

      dataset_str = DATASET_LABEL_PREFIX + class_code + ' ' + ' '.join(str(x) for x in dataset)
      generated_datasets.append(dataset_str)

def retrieve_products(class_code, keywords):
  i = 0
  for keyword in keywords:
    keyword_data = keyword['text']
    keyword_data.strip()
    if keyword_data == '':
      continue
    dataset = retrieve_dataset(keyword_data)
    print('retrieve_dataset() Done : ' + keyword)
    convert_dataset_as_fasttext(class_code, dataset)

def retrieve_dataset(keyword):
  offset = 0
  limit = 100

  dataset = []
  while True:
    products = product_api.get_products_by_keyword(keyword, offset=offset, limit=limit)

    for product in products:
      data = []
      data.append(product['name'])
      # data.extend(product['tags'])
      data.extend(product['cate'])
      data = list(set(data))
      # print('' + str(product['_id']) + ': ' + keyword + ' / ' + str(data))
      dataset.append(data)

    if limit > len(products):
      break
    else:
      offset = offset + limit

  return dataset

def make_dataset():
  classes = text_api.get_classes()
  for class_code in classes:
    retrieve_keywords(class_code['code'])

  shuffle(generated_datasets)
  # print(generated_datasets)

  datasets_total = len(generated_datasets)
  eval_data_count = int(datasets_total / 6)

  print(datasets_total)
  print(datasets_total - eval_data_count)
  print(eval_data_count)

  # datasets for evaluation
  f = open("text_classification_model.eval", 'w')
  for i in range(0, eval_data_count):
    f.write(generated_datasets[i] + '\n')
  f.close()

  # datasets for training
  f = open("text_classification_model.train", 'w')
  for i in range(eval_data_count, datasets_total):
    f.write(generated_datasets[i] + '\n')
  f.close()

  print('Generating dataset Done !!')

def print_model_results(result):
    print("Number of examples for test: " + str(result.nexamples))
    print("P@{}\t{:.3f}".format(1, result.precision))
    print("R@{}\t{:.3f}".format(1, result.recall))

def make_model():
  log.info('make_model')

  train_data = TEXT_CLASSIFICATION_MODEL + '.train'
  valid_data = TEXT_CLASSIFICATION_MODEL + '.eval'

  model = fasttext.supervised(train_data, TEXT_CLASSIFICATION_MODEL, epoch=25, lr=1.0)
  result = model.test(valid_data)

  print_model_results(result)

def start():
  try:
    # make_dataset()
    make_model()

    # predict_test()

    save_model_to_storage()

  except Exception as e:
    log.error(str(e))

if __name__ == '__main__':
  try:
    log.info('Start bl-text-classification-modeler')
    start()
  except Exception as e:
    log.error('main; ' + str(e))
    delete_pod()
