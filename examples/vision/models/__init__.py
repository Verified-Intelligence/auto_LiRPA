from models.example_resnet import model_resnet
from models.example_feedforward import *

Models = {
    'mlp_3layer': mlp_3layer,
    'cnn_4layer': cnn_4layer,
    'cnn_6layer': cnn_6layer,
    'cnn_7layer': cnn_7layer,
    'cnn_7layer_alt': cnn_7layer_alt,
    'resnet': model_resnet,
}
