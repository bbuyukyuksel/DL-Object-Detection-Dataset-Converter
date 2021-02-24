from tqdm import tqdm
import termcolor
import io

class CSV:
    annotations = {}

    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.lines = f.readlines()[1:]

    def parse_csv(self):
        return list(map(lambda x: {
            "filename": x.split(',')[0],
            "width": x.split(',')[1],
            "height": x.split(',')[2],
            "class": x.split(',')[3],
            "bbox": x.split(',')[4:]
        }, self.lines))

    def get_filenames(self):
        return list(set(map(lambda x: x["filename"], self.parse_csv())))

    def get_classnames(self):
        return list(set(map(lambda x: x["class"], self.parse_csv())))

    def get_class_counts(self, data:list):
        counts = {x:0 for x in self.get_classnames()}

        for d in data:
            counts[d["class"]] += 1

        return counts

    def process(self):
        csv = self.parse_csv()
        filenames = self.get_filenames()
        termcolor.cprint("Generating annotation dict..", "yellow")
        for filename in tqdm(filenames):
            if filename not in self.annotations.keys():
                self.annotations[filename] = {}
            
            data = list(filter(lambda x: x["filename"] == filename, csv))

            self.annotations[filename]["count"] = self.get_class_counts(data)
            self.annotations[filename]["data"] = data
        
        return self.annotations

    def calculate(self, dataset:dict):
        counts = {x:0 for x in self.get_classnames()}
        for filename in dataset.keys():
            for label in dataset[filename]["count"].keys():
                #print("Label", label, "Current", counts[label], "Data", dataset[filename]["count"][label])
                counts[label] += dataset[filename]["count"][label]
        return counts

    def calculate_loss(self, counts:dict):
        total_class_name = len(counts.keys())
        total_class_num = sum(obj_class for obj_class in counts.values())
        each_objects_rate = total_class_num // total_class_name

        error = sum([((obj_num - each_objects_rate)/total_class_num)**2 for obj_num in counts.values()])
        return error

    @classmethod
    def write(cls, annotations:dict, file:io.TextIOWrapper):
        file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
        for filename in tqdm(annotations.keys()):
            for obj in annotations[filename]["data"]:
                obj_image_filename = obj["filename"]
                obj_width = obj["width"]
                obj_height = obj["height"]
                obj_class = obj["class"]
                obj_bbox = obj["bbox"]

                str_line = f"{obj_image_filename},{obj_width},{obj_height},{obj_class}," + ",".join(obj_bbox)                    
                file.write(str_line)