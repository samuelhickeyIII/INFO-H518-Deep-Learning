import os
import yaml
import logging.config

def setup_logging():
    """
    Sets up the project logging using a .yaml file for logger configs
    """
    path: str = os.path.join(
        os.getcwd(),
        'Assignments',
        'A1',
        'log',
        'logConfig.yaml'
    )
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs.')