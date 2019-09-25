from keras.models import model_from_json


def load_model(model_path, weight_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weight_path)
    return model
