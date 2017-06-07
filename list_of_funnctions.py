
class Network:


    def get_layer_list(self):
        '''returns a list of layers names
        /in keras every layer of a model has a name by
        which it can be called like in the getactivations function/
        exists in torch? '''
        pass

    def get_layer_output_shape(self, layername):
        '''Give the shape of the output
        of the given layer'''
        return lshape


    def get_activations(self, layername, inputsample):
        '''Gives activations values of the network/model
        for a given layername and an input (inputsample)'''

        return intermediate_output

    def get_layer_weights(self, layername):
        '''Returns weights INCOMING to the
        layer (layername) of the model
        shape of the weights variable should be
        coherent with the get_layer_output_shape function'''
        return weights

    def get_layer_info(self, layername):
        '''Get information on the given layer.
        '''
        return { 'name': layername }


import keras
from keras.models import Model, load_model


class KerasNetwork(Network):

    def __init__(self, modelfilename):
        '''Create a KerasNetwork.
        '''
        print("KerasNetwork: Using keras version: {}".format(keras.__version__))
        self.model = load_model(modelfilename)


    def get_layer_list(self):
        return [ _['config']['name'] for _ in self.model.get_config() ]

    def get_activations(self, layername, input):
        intermediate_layer_model = Model(input=self.model.input,
                                         output=self.model.get_layer(layername).output)
        intermediate_output = intermediate_layer_model.predict(input)
        return intermediate_output

    def get_layer_info(self, layername):
        info = super().get_layer_info(layername)
        layer = self.model.get_layer(layername)
        return info
