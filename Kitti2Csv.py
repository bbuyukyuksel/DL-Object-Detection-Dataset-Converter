import cv2
import glob
import os

from tqdm import tqdm

class Kitti2Csv:
    
    def __init__(self):
        pass

    def try_parse_to_int(self, value:str) -> int:
        return int(value.split('.')[0])

    def parse_annotation(self, file_annotation:str) -> list:
        with open(file_annotation, 'r') as f:
            return list(map(lambda x: {
                "class": x.split(' ')[0],
                "bbox": tuple(map(self.try_parse_to_int, x.split(' ')[4:8])),
                "annotation_filename": file_annotation,
                "image_filename": self.get_image_by_annotation_file(file_annotation),

            }, f.readlines()))
    
    def manipulate_extension_by_annotation_file(self, file_annotation:str, extension:str):
        val = file_annotation.split(".")
        return ".".join(val[:-1]) + "." + extension

    def get_image_extension_by_annotation_file(self, file_annotation:str):
        filename = self.manipulate_extension_by_annotation_file(file_annotation, "*")
        filenames = list(filter(lambda x: ".txt" not in x, glob.glob(filename)))
        assert len(filenames) > 0, Exception("Could not find annotation file in directory!")
        return filenames[0].split('.')[-1]
        
    def get_image_by_annotation_file(self, file_annotation:str):
        return self.manipulate_extension_by_annotation_file(file_annotation, 
                        self.get_image_extension_by_annotation_file(file_annotation)
                )

    def debug_visualize_by_annotation_file(self, file_annotation:str, colormap:dict):
        image_file_path = self.get_image_by_annotation_file(file_annotation)
        image = cv2.imread(image_file_path)

        for obj in annotations:
            cv2.rectangle(image, obj["bbox"][:2], obj["bbox"][2:], colormap[obj["class"]], 1)

        cv2.imshow("Annotation", image)
        cv2.waitKey(0)

    def image_details_by_annotation_file(self, file_annotation:str):
        image_file_path = self.get_image_by_annotation_file(file_annotation)
        image = cv2.imread(image_file_path)

        return {
            "height": image.shape[0],
            "width": image.shape[1]
        }

    class CSV:
        classes = {}
        filename = ""
        def __init__(self, filename):
            self.filename = filename
            self.__create()


        def __create(self):
            with open(self.filename, 'w') as f:
                f.write(f'filename,width,height,class,xmin,ymin,xmax,ymax\n')

        def write(self, annotations, image_details):
            with open(self.filename, 'a+') as f:
                for obj in annotations:
                    obj_class = obj["class"]
                    obj_image_filename = os.path.basename(obj["image_filename"])
                    obj_bbox = obj["bbox"]

                    # Count object class
                    if obj_class not in self.classes.keys():
                        self.classes[obj_class] = 0
                    self.classes[obj_class] += 1

                    str_line = f"{obj_image_filename},{image_details['width']},{image_details['height']},{obj_class}," + ",".join(tuple(map(lambda x: str(x), obj_bbox)))                    
                    f.write(str_line + "\n")
        
        def get_class_counts(self):
            return self.classes
        

def main(annotations_folder:str, csv_filename:str):
    # Example
    '''
    ## Find Annotations '.txt' files in directory
    file_annotations = glob.glob("kitti_dataset/*.txt")
    print(file_annotations[0])
    
    ## Create Labelmap
    colormap = {
        "Mask": (0,255,0),
        "No-Mask":(0,0,255)
    }
    k2csv = Kitti2Csv()
    ## Parse Annotations
    annotations = k2csv.parse_annotation(file_annotations[0])
    print(annotations[0]["class"], annotations[0]["bbox"], annotations[0]["image_filename"])

    ## Visualize Image Annotations
    k2csv.debug_visualize_by_annotation_file(file_annotations[0], colormap=colormap)

    ## Get image details
    image_details = k2csv.image_details_by_annotation_file(file_annotations[0])

    mycsv = k2csv.CSV("test.csv")
    mycsv.write(annotations, image_details)
    '''

    k2csv = Kitti2Csv()
    mycsv = k2csv.CSV(csv_filename)
    for file_annotation in tqdm(glob.glob(f"{annotations_folder}/*.txt")):
        annotations = k2csv.parse_annotation(file_annotation)
        image_details = k2csv.image_details_by_annotation_file(file_annotation)
        mycsv.write(annotations, image_details)
    print(mycsv.get_class_counts())

if __name__ == '__main__':
    import argparse

    usage = f'''
    ------------------------------------------------------------------------------------------------------------------------------------
        # Kitti to CSV
        >> python Kitti2Csv.py --folder "kitti_dataset" --output "dataset.csv"
    ------------------------------------------------------------------------------------------------------------------------------------
    '''

    parser = argparse.ArgumentParser(description="Kitti to CSV Dataset Converter", usage=usage)
    parser.add_argument("-f", "--folder", required=True, help="Annotation Folder Path")
    parser.add_argument("-o", "--output", required=True, help="CSV output dataset name")
    args = parser.parse_args()

    main(annotations_folder=args.folder, csv_filename=args.output)