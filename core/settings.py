import os
import json

class Settings:
    def __init__(self):
        self.filename = "settings.json"  
        self.settings_dict = dict()     

    def save(self):
        pass
        # f = open(self.filename, 'w')
        
        # # Describe output type of the processed video
        # self.settings["output_type"] = "video"
        # # self.settings["output_type"] = "images"

        # # Head Pose Estimation Output
        # # self.settings["hpe"] = "yes"
        # # self.settings["hpe"] = "no"

        # f.write(str(self.settings))
        # f.close()        

    def load(self):    
        f = open(self.filename, 'r')
        data = f.read()
        self.settings_dict = json.loads(data)
        f.close()    
        pass
