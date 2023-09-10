import pandas as pd

class ClientReport():
    def __init__(self):
        pass

    def generateReport(self, filename, fps):
        self.df = pd.read_csv(filename)
        timestamp = self.temporal_segment()
        time = self.extract_timestamp(timestamp, fps)        

    def temporal_segment(self):        
        label_ls = list(self.df['Label'])
        timestamp = []
        start = 0
        end = 0
        current_label = -1
        for index, label in enumerate(label_ls):
            if current_label != label:
                end = index - 1
                timestamp.append((start, end, current_label))
                current_label = label
                start = index

        # If meet end of list
        timestamp.append((start, len(label_ls) - 1, current_label))
        # Remove first item in list 
        timestamp.pop(0)        
        return timestamp        
    
    def extract_timestamp(self, timestamp, fps):        
        time = []        
        for item in timestamp:
            start, end, label = item
            start = self.frame_to_minutes(start, fps)
            end = self.frame_to_minutes(end, fps)
            time.append((start, end, label))

        print(time)
        return time

    def frame_to_minutes(self, frames, fps):
        total_seconds = frames / fps
        minutes = str(int(total_seconds / 60))
        seconds = str(int(total_seconds % 60))
        time = minutes + ":" + seconds
        return time

    def summary(self):
        pass
    
class ClientReportBinary(ClientReport):
    def __init__(self):
        ClientReport.__init__(self)

    def generateReport(self, filename, fps):
        self.df = pd.read_csv(filename)
        timestamp = self.temporal_segment()
        time = self.extract_timestamp(timestamp, fps)  
        self.summary()

    def summary(self):
        label_df = self.df['Label']
        total_frame = len(label_df)
        frame_0 = len(label_df[(label_df == 0)])
        frame_1 = len(label_df[(label_df == 1)])
        print("Frame Not Expression :", '{:.2f}'.format((frame_0 / total_frame) * 100), "%")
        print("Frame Expression     :", '{:.2f}'.format((frame_1 / total_frame) * 100), "%")
            


if __name__ == "__main__":
    filename = r"D:\CodeProject2\SKRIPSI_FINAL\reports\BUKU\binary\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch\S50_binary.csv"
    clientReport = ClientReportBinary()
    clientReport.generateReport(filename, 25)

