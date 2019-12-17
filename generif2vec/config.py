import configparser
import os

from generif2vec import __project__, __version__
from loguru import logger

# create config
config = configparser.ConfigParser()

# create config directory if it doesn't exist
config_directory = os.path.join(os.environ.get('HOME'), '.{}'.format(__project__))
project_directory = os.path.abspath(__project__)
project_data_dir = os.path.join(project_directory, 'data')
try:
    os.makedirs(config_directory)
except FileExistsError:
    pass

# create data directory if it doesn't exist
data_directory = os.path.join(config_directory, 'data')
try:
    os.makedirs(data_directory)
except FileExistsError:
    pass

# if generif2vec.ini doesnt exist make one
logger.info('checking if config file exists: {}'.format(config_directory))
if not os.path.isfile(os.path.join(config_directory, '{}.ini'.format(__project__))):
    config = configparser.ConfigParser()
    config['entrez'] = {
            'user_name': '',
            'api_key': '',
        }
    config['data'] = {
        'data_directory': project_data_dir,
    }

    with open(os.path.join(config_directory, '{}.ini'.format(__project__)), 'w') as configfile:
        logger.info('writing config file to: {} '.format(config_directory))
        config.write(configfile)

# log project and version
logger.info('{} {}'.format(__project__, __version__))

# read config
config_file = os.environ.get(
    '{}_CONFIG'.format(__project__.upper()),
    os.path.join(config_directory, '{}.ini'.format(__project__),)
)
config.read(config_file)
logger.info('Using configuration file: {}'.format(config_file))

DATA_DIRECTORY = config.get('data', 'data_directory')
ENTREZ_USER_NAME = config.get('entrez', 'user_name')
ENTREZ_API_KEY = config.get('entrez', 'api_key')
