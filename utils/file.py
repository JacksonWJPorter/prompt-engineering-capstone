import yaml
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from os import path

def load_dictionary(category):
    file_name = f'model_configs/{category}.yaml'
    file_name = path.normpath(file_name)
    with open(file_name, 'r') as f:
        category_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    return category_dictionary

def load_dictionary_agentic(category):
    file_name = f'model_configs_agentic/{category}.yaml'
    file_name = path.normpath(file_name)
    with open(file_name, 'r') as f:
        category_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    return category_dictionary

def load_safety_settings(file_path):
    file_path = path.normpath(file_path)
    with open(file_path, "r") as yamlfile:
        settings = yaml.safe_load(yamlfile)

    safety_settings = {}
    for setting in settings["safety_settings"]:
        category = getattr(HarmCategory, setting["category"], None)
        threshold = getattr(HarmBlockThreshold, setting["threshold"], None)
        
        if category is not None and threshold is not None:
            safety_settings[category] = threshold
        else:
            raise ValueError(f"Invalid category or threshold: {setting}")

    return safety_settings
