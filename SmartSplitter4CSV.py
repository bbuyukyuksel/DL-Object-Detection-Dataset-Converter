from tqdm import tqdm
from random import shuffle
import json
import termcolor
import os
import time
import io

from CSV import CSV

def SmartSplitter(filename, train, iteration):
    mycsv = CSV(filename)
    annotations = mycsv.process()

    filenames = mycsv.get_filenames()
    to = int(len(filenames) * train)

    register = {}
    best_error = 9999.99
    for i in tqdm(range(iteration)):
        shuffle(filenames)

        train_filenames = filenames[:to]
        test_filenames  = filenames[to:]

        train_dataset = {x:annotations[x] for x in train_filenames}
        test_dataset  = {x:annotations[x] for x in test_filenames }
    
        train_counts = mycsv.calculate(train_dataset)
        test_counts  = mycsv.calculate(test_dataset )

        train_error = mycsv.calculate_loss(train_counts)
        test_error  = mycsv.calculate_loss(test_counts )
        error = ((4 * train_error) + (1 * test_error)) / 5 

        if error < best_error:
            best_error = error

            termcolor.cprint(f"\nBest Values, Error: {best_error} || {train_counts} || {test_counts}", "green")

            register = {
                "train_counts": train_counts,
                "train_filenames": train_filenames,
                "train_dataset": train_dataset,

                "test_counts": test_counts,
                "test_filenames": test_filenames,
                "test_dataset": test_dataset,

                "error":error,
            }

            with open('register.json', 'w') as f:
                json.dump(register, f, indent=4)
            
def Export(image_folder_path, export_path):
    import shutil

    def generate_csv(selection, data, image_folder_path, export_path):
        dst_dir = os.path.join(export_path, selection)
        os.makedirs(dst_dir, exist_ok=True)
        train_filenames = data[f"{selection}_filenames"]
        termcolor.cprint(f"Copying from {image_folder_path} to {dst_dir}", "yellow")
        for filename in tqdm(train_filenames):
            src = os.path.join(image_folder_path, filename)
            dst = os.path.join(dst_dir, filename)
            shutil.copyfile(src, dst)

        ## Generate CSV
        csv_selection = os.path.join(export_path, f"{selection}.csv")
        with open(csv_selection, "w") as file:
            termcolor.cprint("Writing CSV File ...", "yellow")
            CSV.write(data[f"{selection}_dataset"], file)

        ## Check Class Counts
        csv_test = CSV(csv_selection)
        data = csv_test.parse_csv()
        counts = csv_test.get_class_counts(data)
        termcolor.cprint(counts, "green")        
        

    ## Check Paths
    assert os.path.exists('register.json'), Exception("Register.json not exists!")
    assert os.path.exists(image_folder_path), Exception("Image foulder path not exists!")
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    os.makedirs(export_path, exist_ok=True)

    data = None
    with open('register.json', 'r') as f:
        data = json.load(f)
    
    # Create Train Dir
    generate_csv("train", data, image_folder_path, export_path)
    generate_csv("test",  data, image_folder_path, export_path)
    

if __name__ == '__main__':
    usage = f'''
    ------------------------------------------------------------------------------------------------------------------------------------
        # SmartSplitter
        {termcolor.colored("**required: dataset.csv", "red")}
        >> python SmartSplitter4CSV.py --trainsplit 0.8 --iteration 200000 --csvfilename "out/dataset.csv" --imagefolderpath "datasets/kitti"

        # Export
        {termcolor.colored("**required: register.json", "red")}
        >> python SmartSplitter4CSV.py --csvfilename "out/dataset.csv" --imagefolderpath "datasets/kitti" --export True 
    ------------------------------------------------------------------------------------------------------------------------------------
    '''
    import argparse

    parser = argparse.ArgumentParser(description="CSV Dataset Smart Splitter", usage=usage)

    #SmartSplitter Process Params
    parser.add_argument("-s", "--trainsplit",  required=False, type=float, default=0.8,             help="train split rate")
    parser.add_argument("-i", "--iteration",   required=False, type=int,   default=1000,            help="iteration")
    parser.add_argument("-c", "--csvfilename", required=True,  type=str,                            help="CSV filename")


    #Export Params
    parser.add_argument("-e", "--export",           required=False, help="export register.json into dataset")
    parser.add_argument("-p", "--imagefolderpath",  required=True,  help="image folder path")
    


    args = parser.parse_args()

    if args.export:
        image_folder_path = args.imagefolderpath
        export_path = "smartsplitter_csv_dataset"

        Export(image_folder_path, export_path)
    else:
        train = args.trainsplit
        filename = args.csvfilename
        iteration = args.iteration

        SmartSplitter(filename, train, iteration)
    