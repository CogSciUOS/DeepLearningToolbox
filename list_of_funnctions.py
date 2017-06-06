
def get_layers_list(model):
    '''returns a list of layers names
        /in keras every layer of a model has a name by
        which it can be called like in the getactivations function/
        exists in torch? '''
    return list



def get_layer_output_shape(model, layername):
    '''Give the shape of the output
        of the given layer'''
    return lshape





def get_activations(model,layername,inputsample):
    '''Gives activations values of the network/model
        for a given layername and an input (inputsample)'''

    return intermediate_output




def get_layer_weights(model,layername):
    '''Returns weights INCOMING to the
       layer (layername) of the model
       shape of the weights variable should be
       coherent with the get_layer_output_shape function'''
    return weights
