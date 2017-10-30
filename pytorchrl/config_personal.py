
import os.path as osp
import os

USE_GPU = False

AWS_REGION_NAME = 'us-east-2'

if USE_GPU:
    DOCKER_IMAGE = 'dementrock/rllab3-shared-gpu'
else:
#    DOCKER_IMAGE = 'dwicke/jump:docker'
    DOCKER_IMAGE = 'dwicke/parameterized:latest'

DOCKER_LOG_DIR = '/tmp/expt'

AWS_S3_PATH = 's3://pytorchrl/pytorchrl/experiments'

AWS_CODE_SYNC_S3_PATH = 's3://pytorchrl/pytorchrl/code'

ALL_REGION_AWS_IMAGE_IDS = {
    'ap-northeast-1': 'ami-002f0167',
    'ap-northeast-2': 'ami-590bd937',
    'ap-south-1': 'ami-77314318',
    'ap-southeast-1': 'ami-1610a975',
    'ap-southeast-2': 'ami-9dd4ddfe',
    'eu-central-1': 'ami-63af720c',
    'eu-west-1': 'ami-41484f27',
    'sa-east-1': 'ami-b7234edb',
    'us-east-1': 'ami-83f26195',
    'us-east-2': 'ami-66614603',
    'us-west-1': 'ami-576f4b37',
    'us-west-2': 'ami-b8b62bd8'
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = 'g2.2xlarge'
else:
    AWS_INSTANCE_TYPE = 'c4.2xlarge'

ALL_REGION_AWS_KEY_NAMES = {
    "us-east-2": "pytorchrl-us-east-2",
    "us-east-1": "pytorchrl-us-east-1"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.1'

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY', None)

AWS_ACCESS_SECRET = os.environ.get('AWS_ACCESS_SECRET', None)

AWS_IAM_INSTANCE_PROFILE_NAME = 'pytorchrl'

AWS_SECURITY_GROUPS = ['pytorchrl-sg']

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-2": [
        "sg-18009370"
    ],
    "us-east-1": [
        "sg-46308b34"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    '.git',
    'data/local',
    'data/s3',
    'data/video',
    'src',
    '.idea',
    '.pods',
    'tests',
    'examples',
    'docs',
    '.idea',
    '.DS_Store',
    '.ipynb_checkpoints',
    'blackbox',
    'blackbox.zip',
    '*.pyc',
    '*.ipynb',
    'scratch-notebooks',
    'conopt_root',
    'private/key_pairs',
]

FAST_CODE_SYNC = True

