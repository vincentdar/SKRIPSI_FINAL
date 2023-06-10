import os
import json

class Settings:
    def __init__(self):
        self.filename = "settings.json"  
        self.settings_dict = dict()     

    def default_config(self):                
        # Describe output type of the processed video
        # ['Images', 'Clip', 'Video']
        # self.settings["output_type"] = "Images"
        self.settings_dict["output_type"] = "Video"        
        # self.settings["output_type"] = "Clip"

        # Binary Or Multiclass
        self.settings_dict["classification"] = "binary"
        # self.settings["classification"] = "multiclass"        

        # Serializing json          
        with open("settings.json", "w") as outfile:
            json.dump(self.settings_dict, outfile)        
        

    def load(self):    
        try:
            f = open(self.filename, 'r')
            data = f.read()
            self.settings_dict = json.loads(data)
            print(self.settings_dict["output_type"])
            print(self.settings_dict["classification"])
            f.close()    
        except Exception as e:
            print("Config file corrupt.. set to default config")
            self.default_config()
            self.load()
