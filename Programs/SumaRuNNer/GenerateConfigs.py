import os

import keras
from keras import backend as K
import pickle
import sys
import yaml


def generate_agg_config(folder_configs):
    """
    Generates config file for aggregator

    :return: None
    """
    
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)
    config_file = os.path.join(folder_configs, 'config_agg.yml')

    connection = {
        'name': 'FlaskConnection',
        'path': 'ffl_framework.connection.flask_connection',
        'info': {
            'ip': '127.0.0.1',
            'port': 5000
        },
        'synch': False
    }

    data = {
        'name': 'SumaDataHandler',
        'path': 'Programs.SumaRuNNer.DataHandler',
        'info': {
                'X_train_file': "./Vendor/Enc_Train_Vendor_docs.npy",
                'y_train_file': "./Vendor/Enc_Train_Vendor_labels.npy",
                'X_test_file': "./Vendor/Enc_Test_Vendor_docs.npy",
                'y_test_file': "./Vendor/Enc_Test_Vendor_labels.npy"
        }
    }

    fusion = {
        'name': 'SGDFusionHandlerHandler',
        'path': 'ffl_framework.aggregator.fusion.fusion_handler'
    }

    hyperparams = {
        'global': {'epochs': 5,
                   'termination_accuracy': 0.98
                   },
        'local': {'training': {'epochs': 2, 'batch_size': 32},
                  'optimizer': {'lr': 0.001}
                  }
    }
    protocol_handler = {
        'name': 'ProtoHandler',
        'path': 'ffl_framework.aggregator.protohandler.proto_handler'
    }
    content = {
        'connection': connection,
        'data': data,
        'fusion': fusion,
        'hyperparams': hyperparams,
        'protocol_handler': protocol_handler
    }

    with open(config_file, 'w') as outfile:
        yaml.dump(content, outfile)

    print('Finished generating config file for aggregator. Files can be found in: ',
          os.path.abspath(config_file))


def generate_party_config_file(party, i):

    # Model location
    fname = "./Models/SummaRuNNer_spec.h5"

    # Generate model spec:
    spec = {'model_name': 'keras-cnn',
            'compiled_model': fname}



    # Now generate automatically config files for all parties:  
    folder_configs = party+"/configs"
    
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)
    config_file = os.path.join(folder_configs, 'config_party.yml')

    
    f_spec = os.path.join(folder_configs, 'model_spec.pickle')
    with open(f_spec, 'wb') as f:
        pickle.dump(spec, f)
        
    config_file = os.path.join(
        folder_configs, 'config_party' + str(i) + '.yml')
    model = {
        'name': 'KerasFFLModel',
        'path': 'ffl_framework.model.keras_ffl_model',
        'spec': os.path.join(folder_configs, 'model_spec.pickle')
    }

    connection = {
        'name': 'FlaskConnection',
        'path': 'ffl_framework.connection.flask_connection',
        'info': {
            'ip': '127.0.0.1',
            'port': 8085 + i
        },
        'synch': False
    }
    data = {
        'name': 'SumaDataHandler',
        'path': 'Programs.SumaRuNNer.DataHandler',
        'info': {
            'X_train_file': "./{}/Enc_Train_{}_docs.npy".format(party, party),
            'y_train_file': "./{}/Enc_Train_{}_labels.npy".format(party, party),
            'X_test_file': "./{}/Enc_Test_{}_docs.npy".format(party, party),
            'y_test_file': "./{}/Enc_Test_{}_labels.npy".format(party, party)
        }
    }
    protocol_handler = {
        'name': 'ProtocolHandlerPlainFL',
        'path': 'ffl_framework.party.protocol_handler_plain_fl'
    }
    aggregator = {
        'ip': '127.0.0.1',
        'port': 5000
    }
    content = {
        'connection': connection,
        'data': data,
        'model': model,
        'protocol_handler': protocol_handler,
        'aggregator': aggregator
    }

    with open(config_file, 'w') as outfile:
        yaml.dump(content, outfile)

    print('Finished generating config file. Files can be found in: ',
          os.path.abspath(os.path.join(folder_configs, 'config_party')))


if __name__ == '__main__':
    nb_parties = int(sys.argv[1])
    generate_party_config_file("Vendor", 0)
    generate_party_config_file("Buyer1", 1)
    generate_party_config_file("Buyer2", 2)
    generate_agg_config("Aggregator/configs")
