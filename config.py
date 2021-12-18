class DefaultConfig(object):
    # path of images directory
    dir_path = '../data_set/IMAGES'

    # path of xml files directory
    xml_path = '../data_set/ANNOTATIONS'

    num_epochs = 30
    batch_size = 2,
    shuffle = True,
    num_workers = 2,

    lr = 0.01,
    momentum = 0.9,
    weight_decay = 0.0005
