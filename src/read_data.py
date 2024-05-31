import json

class MTDataReader:
    # 读取machine translation数据，输入文件格式为json，每行为一个json对象，包含源语言和目标语言的句子 
    def __init__(self, filename, lan='en-tr'):
        self.filename = filename
        self.src = lan.split('-')[0]
        self.tgt = lan.split('-')[1]

    def read_data(self):
        data = []
        with open(self.filename, 'r') as file:
            for line in file:
                line = json.loads(line)['translation']
                data.append((line[self.src], line[self.tgt]))
        return data

class SummaryDataReader:
    def __init__(self, filename):
        self.filename = filename

    def read_data(self):
        data = []
        with open(self.filename, 'r') as file:
            for line in file:
                line_dict = json.loads(line)
                summary = line_dict['summary']
                document = line_dict['document']
                # id_ = line_dict['id']
                data.append((document, summary))
        return data
    
# Example usage
reader = SummaryDataReader('/root/autodl-tmp/xy/benchmark-ralm/datasets/summarization/multinews/multinews_test.json')
summ_data = reader.read_data()
print(summ_data[0])