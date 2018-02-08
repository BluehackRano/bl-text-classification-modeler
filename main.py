from __future__ import print_function

import os
from util import s3
import redis
import fasttext
from bluelens_log import Logging
from bluelens_spawning_pool import spawning_pool
from stylelens_dataset.texts import Texts
from stylelens_product.products import Products
from stylelens_product.models import Models
import codecs

from random import shuffle
import re

SPAWN_ID = os.environ['SPAWN_ID']
RELEASE_MODE = os.environ['RELEASE_MODE']

REDIS_SERVER = os.environ['REDIS_SERVER']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']
REDIS_CRAWL_VERSION = 'bl:crawl:version'
REDIS_CRAWL_VERSION_LATEST = 'latest'

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')
AWS_MODEL_BUCKET = 'bluelens-style-model'
AWS_BUCKET_CLASSIFICATION_TEXT_PATH = 'classification/text/' + RELEASE_MODE + '/'

def get_latest_crawl_version():
  value = rconn.hget(REDIS_CRAWL_VERSION, REDIS_CRAWL_VERSION_LATEST)
  return value

# REDIS_PRODUCT_TEXT_MODEL_PROCESS_QUEUE = 'bl:product:text:model:process:queue'

options = {
  'REDIS_SERVER': REDIS_SERVER,
  'REDIS_PASSWORD': REDIS_PASSWORD
}
log = Logging(options, tag='bl-text-classification-modeler')
rconn = redis.StrictRedis(REDIS_SERVER, decode_responses=True, port=6379, password=REDIS_PASSWORD)

storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

text_api = Texts()
product_api = Products()
model_api = Models()
PRODUCT_MODELS_TYPE = 'text-classification'

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

def save_eval_to_storage():
  log.info('save_eval_to_storage')

  eval_file_name = TEXT_CLASSIFICATION_MODEL + '.eval'

  file = os.path.join(os.getcwd(), eval_file_name)
  try:
    return storage.upload_file_to_bucket(AWS_MODEL_BUCKET, file,
                                         AWS_BUCKET_CLASSIFICATION_TEXT_PATH + eval_file_name)
  except:
    log.error('upload error')
    return None

def save_model_to_storage():
  log.info('save_text_model_to_s3_storage')

  model_file_name = TEXT_CLASSIFICATION_MODEL + '.bin'

  file = os.path.join(os.getcwd(), model_file_name)
  try:
    return storage.upload_file_to_bucket(AWS_MODEL_BUCKET, file,
                                         AWS_BUCKET_CLASSIFICATION_TEXT_PATH + model_file_name)
  except:
    log.error('upload error')
    return None

def save_tmp_text_dataset_to_local(text_code, datasets):

  # tmp_text_dataset
  try:
    f = codecs.open("tmp_text_dataset", 'a', 'utf-8')
    for dataset in datasets:
      count = len(dataset)
      dataset_str = DATASET_LABEL_PREFIX + text_code + ' ' + ' '.join(str(x) for x in dataset)
      print(dataset_str)
      f.write(dataset_str + '\n')
  except IOError:
    print('tmp_text_dataset write error !!!')
  finally:
    f.close()

def retrieve_keywords_with_text_code(text_code):
  offset = 0
  limit = 100
  while True:
    keywords = text_api.get_texts(text_code, offset=offset, limit=limit)
    retrieve_products(text_code, keywords)

    if limit > len(keywords):
      break
    else:
      offset = offset + limit

def convert_dataset_as_fasttext(text_code, datasets):
  # save_tmp_text_dataset_to_local(text_code, datasets)

  for dataset in datasets:
    count = len(dataset)
    if count > 5:
      count = 5
    for n in range(0, count):
      shuffle(dataset)

      dataset_str = DATASET_LABEL_PREFIX + text_code + ' ' + ' '.join(str(x.strip()) for x in dataset)
      generated_datasets.append(dataset_str)

  print('convert_dataset_as_fasttext DONE')

def retrieve_products(text_code, keywords):
  for keyword in keywords:
    keyword_data = keyword.get('text')
    keyword_data.strip()
    if keyword_data == '':
      continue
    log.info('retrieve_products_from_db_and_update() : ')
    log.info(keyword.get('text').encode('utf-8'))
    dataset = retrieve_products_from_db_and_update(keyword_data)

    # print(str(text_code) + ' : ' + keyword_data + ' / ')

    convert_dataset_as_fasttext(text_code, dataset)

def retrieve_products_from_db_and_update(keyword):
  datasets = []
  while True:
    products = product_api.get_products_by_keyword(keyword,
                                                   only_text=True, is_processed_for_text_class_model=False)

    if 0 == len(products):
      break

    for product in products:
      data = []

      name = product.get('name')
      cate = product.get('cate')
      tags = product.get('tags')

      # print('name: {name} / cate: {cate} / tags: {tags}'.format(**product))

      if name is not None:
        data.append(name)
      if cate is not None:
        data.extend(cate)
      # if tags is not None:
      #   data.extend(tags)

      data = list(set(data))

      datasets.append(data)

      product['is_processed_for_text_class_model'] = True

    product_api.update_products(products)

  return datasets

def make_dataset():
  log.info('make_dataset')

  classes = text_api.get_classes()
  for text_code in classes:
    retrieve_keywords_with_text_code(text_code['code'])

  shuffle(generated_datasets)

  datasets_total = len(generated_datasets)
  eval_data_count = int(datasets_total / 6)

  print('total : ' + str(datasets_total))
  print('training : ' + str(datasets_total - eval_data_count))
  print('evaluation : ' + str(eval_data_count))

  i = 0
  # datasets for evaluation
  try:
    f = codecs.open(TEXT_CLASSIFICATION_MODEL + '_temp.eval', 'w', 'utf-8')
    for i in range(0, eval_data_count):
      f.write(generated_datasets[i] + '\n')
  except IOError:
    print('eval_file write error : ' + str(i) + '' + generated_datasets[i].unicode('utf-8'))
  finally:
    f.close()

  i = 0
  # datasets for training
  try:
    f = codecs.open(TEXT_CLASSIFICATION_MODEL + '_temp.train', 'w', 'utf-8')
    for i in range(eval_data_count, datasets_total):
      f.write(generated_datasets[i] + '\n')
  except IOError:
    print('train_file write error : ' + str(i) + '' + generated_datasets[i].unicode('utf-8'))
  finally:
    f.close()

  log.info('temp dataset generated')

  os.system('cat ' + TEXT_CLASSIFICATION_MODEL + '_temp.eval '
                                                 '| sed -e "s/\([.\[\!?,^\'~+&*@#$%=/{}\\"()]\)/ /g" '
                                                 '| tr "[:upper:]" "[:lower:]" > ' + TEXT_CLASSIFICATION_MODEL + '.eval')
  os.system('cat ' + TEXT_CLASSIFICATION_MODEL + '_temp.train '
                                                 '| sed -e "s/\([.\[\!?,^\'~+&*@#$%=/{}\\"()]\)/ /g" '
                                                 '| tr "[:upper:]" "[:lower:]" > ' + TEXT_CLASSIFICATION_MODEL + '.train')
  log.info('dataset normalized !')


def print_model_results(result):
    print("Number of examples for test: " + str(result.nexamples))
    print("P@{}\t{:.3f}".format(1, result.precision))
    print("R@{}\t{:.3f}".format(1, result.recall))

def make_model():
  log.info('make_model')

  train_data = TEXT_CLASSIFICATION_MODEL + '.train'
  valid_data = TEXT_CLASSIFICATION_MODEL + '.eval'

  model = fasttext.supervised(train_data, TEXT_CLASSIFICATION_MODEL, epoch=25, lr=1.0, word_ngrams=2, bucket=5000000)
  result = model.test(valid_data)

  print_model_results(result)

""""""""
# predict test
""""""""
def print_results(results):
    for result in results:
        for data in result:
            print("Class Code : " + data[0] + " / proba : " + str(data[1]))

def predict_test():
  model_data = TEXT_CLASSIFICATION_MODEL + '.bin'

  model = fasttext.load_model(model_data)
  test_data = [
                '롱패딩',
                '블리블링 블라우스, 여신 블라우스',
                'v넥 허니니트 니트 긴팔 v 허니니트 knit 반가다 브이넥 (니트)#12게이지#루즈핏#여리여리',
                '큐트체크미니스커트(밴딩) 버클 데일리 벨트미니스커트(도톰, A라인)  # 벨트탈부착#속바지 체크 12-2김유난 핫바디 미니스커트  #치마바지',
                '겨울원피스 베이비돌 원피스 김다은 11-2김세희 12-1김세희 꽃원피스 걸스 벨벳뷔스티에OPS 미니원피스',
                '모찌 브이넥 가디건',
                '스틱 실버 귀걸이',
                '항공점퍼랑 패딩이랑 (양면패딩)',
               ]
  normalized_test_data = []

  for data in test_data:
    normalized = re.sub('[^가-힝A-Za-z0-9]+', ' ', data)
    lower = normalized.lower()

    normalized_test_data.append(lower)

  results = model.predict_proba(normalized_test_data)
  print_results(results)
""""""""
# predict test ends.
""""""""

def start():
  try:
    version_id = get_latest_crawl_version()
    # doing
    model = {
      'status': 'doing'
    }
    model_api.update_model(PRODUCT_MODELS_TYPE, version_id, model)
    log.info('Doing : bl-text-classification-modeler')

    make_dataset()
    make_model()
    save_model_to_storage()
    save_eval_to_storage()

    predict_test()

    # done
    model['status'] = 'done'
    model_api.update_model(PRODUCT_MODELS_TYPE, version_id, model)
    log.info('Done : bl-text-classification-modeler')

    # if (rconn.blpop([REDIS_PRODUCT_TEXT_MODEL_PROCESS_QUEUE])):
    #   log.info('SUCCESS : bl-text-classification-modeler')

  except Exception as e:
    log.error(str(e))

if __name__ == '__main__':
  try:
    log.info('Start bl-text-classification-modeler')
    start()
  except Exception as e:
    log.error('main; ' + str(e))
    # delete_pod()
