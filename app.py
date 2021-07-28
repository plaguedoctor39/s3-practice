import json
import os

import boto3
from secrets import access_key, secret_access_key
from train import train

CLIENT = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
BUCKET_NAME = 's3-practice-buck'


def get_file(name):
    CLIENT.download_file(BUCKET_NAME, f'datasets/{str(name)}', 'data/' + str(name))
    print('Done')


def post_file(name, file_key):
    upload_file_name = name
    upload_file_key = str(file_key) + '/' + str(upload_file_name)
    CLIENT.upload_file(upload_file_name, BUCKET_NAME, upload_file_key)
    print('Done')


def model_training(data_conf, json_data):
    with open(json_data) as json_file:
        model_conf = json.load(json_file)
    train(data_conf, model_conf)
    os.chdir('artifacts/output')
    for file in os.listdir():
        post_file(file, 'artifacts/output')
    print('Uploaded trained model')


def main():
    cmd = input('Enter command -> ')
    if cmd == 'post':
        post_file('diabetes.csv', 'datasets')
    elif cmd == 'get':
        get_file('diabetes.csv')
    elif cmd == 'train':
        model_training('data/diabetes.csv', 'config.json')
    else:
        print('wrong command')


main()
