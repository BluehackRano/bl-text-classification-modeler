from __future__ import print_function

import os
from util import s3
import redis
from bluelens_log import Logging
from bluelens_spawning_pool import spawning_pool
from stylelens_dataset.texts import Texts
from stylelens_product.products import Products

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
log = Logging(options, tag='bl-object-classifier')
rconn = redis.StrictRedis(REDIS_SERVER, decode_responses=True, port=6379, password=REDIS_PASSWORD)

storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

text_api = Texts()
product_api = Products()

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

def convert_dataset_as_fasttext(class_code, dataset):
  #Todo: Need to implement by rano@bljuehack.net
  # Example)
  # class_code : '1'
  # datasets[0] : ['aaa', 'bbb', 'ccc', 'ddd]
  # datasets[1] : ['aaa', 'bbbc', 'ccca', 'ddda']
  # datasets[n] :
  log.info('convert_dataset_as_fasttext')

def retrieve_products(class_code, keywords):
  for keyword in keywords:
    dataset = retrieve_dataset(keyword['text'])
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
      print('' + str(product['_id']) + ':' + str(data))
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

def make_model():
  #Todo: Need to implement by rano@bljuehack.net
  log.info('make_model')

def start():
  try:
    make_dataset()
    make_model()
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
