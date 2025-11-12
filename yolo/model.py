import yaml
import os
import cv2
from ultralytics import YOLO
from argparse import ArgumentParser, BooleanOptionalAction
from ultralytics.utils.plotting import plot_results

# arguments from the command line
parser = ArgumentParser(description='Bustine Detection YOLOv11')
parser.add_argument('--data', type=str, help='data.yaml path (can be set also in the congif.yaml file)')
parser.add_argument('--output', type=str, help='output directory for the model')
parser.add_argument('--model', type=str, help='model path')
parser.add_argument('--train', type=bool, help='start the training on the dataset', action=BooleanOptionalAction)
parser.add_argument('--test', type=bool, help='start the testing on the dataset', action=BooleanOptionalAction)
parser.add_argument('--forward', type=str, help='evaluate the model on a video',)
parser.add_argument('--show_dataset', type=bool, help='show the dataset', action=BooleanOptionalAction)
parser.add_argument('--plot_result', type=str, help='plot the results from the results.csv file')
args = parser.parse_args()

# arguments from the config file
CONFIGS = yaml.safe_load(open('config.yaml'))

# merge conflicting arguments
DATA_FILE = args.data if args.data else CONFIGS['DATA']['YAML_FILE']
OUTPUT_DIR = args.output if args.output else CONFIGS['DATA']['OUTPUT_FOLDER']
MODEL = args.model if args.model else CONFIGS['MODEL']['MODEL_PATH']
NUM_EPOCHS = CONFIGS['TRAIN']['EPOCHS'] if CONFIGS['TRAIN']['EPOCHS'] else 100
IMG_SIZE = CONFIGS['TRAIN']['IMG_SIZE'] if CONFIGS['TRAIN']['IMG_SIZE'] else 640
DEVICE = CONFIGS['MISC']['DEVICE'] if CONFIGS['MISC']['DEVICE'] else None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# show the dataset
if args.show_dataset:
    DATA_CONFIGS = yaml.safe_load(open(DATA_FILE))
    dataset_root_dir = DATA_FILE.split('data.yaml')[0]
    image_dir = os.path.join(dataset_root_dir, 'images')
    label_dir = os.path.join(dataset_root_dir, 'labels')

    image_output_dir = os.path.join(OUTPUT_DIR, 'dataset')
    os.makedirs(image_output_dir, exist_ok=True)

    for subset in ['train', 'val', 'test']:
        image_subset_dir = os.path.join(image_dir, subset)
        label_subset_dir = os.path.join(label_dir, subset)

        for image in os.listdir(image_subset_dir):
            image_path = os.path.join(image_subset_dir, image)
            label_path = os.path.join(label_subset_dir, image.replace('jpg', 'txt'))
            
            # read the image and the label
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape

            with open(label_path, 'r') as f:
                for line in f:
                    _, x_center, y_center, bbox_width, bbox_height = [float(x) for x in line.split(' ')]
                
                    x1 = int((x_center - bbox_width / 2) * img_width)
                    y1 = int((y_center - bbox_height / 2) * img_height)
                    x2 = int((x_center + bbox_width / 2) * img_width)
                    y2 = int((y_center + bbox_height / 2) * img_height)
                    
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            cv2.imwrite(os.path.join(image_output_dir, image), img)
    exit()

if args.plot_result:
    print('Plotting the results')
    plot_results(file=args.plot_result)
    exit()

model = YOLO(
    model=MODEL,
    task='detect',
)

if args.train:
    # model.resume = True
    train_results = model.train(
        data=DATA_FILE,
        epochs=NUM_EPOCHS,
        device=DEVICE,
        # agnostic_nms=True,
        # amp=False,
        # patience=0,
        # weight_decay=0.001,
        dropout=0.8,
        #max_det=1,
        #conf=0.1,
        
        # augmentation
        # hsv_h=0.1,
        # hsv_s=0.9,
        # hsv_v=0.9,
        degrees=0,
        translate=0,
        scale=0,
        shear=0,
        perspective=0,
        # flipud=0.0,
        # fliplr=0.5,
        bgr=0,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        erasing=0,
    )
    # print(train_results)
    exit()

if args.test:
    eval_results = model.val()
    exit()

if args.forward == 'test' or args.forward == 'train' or args.forward == 'val':
    data_path = yaml.safe_load(open(DATA_FILE))
    img_paths = [os.path.join('./datasets/images/', args.forward, img) for img in os.listdir(os.path.join('./datasets', data_path[args.forward]))]
    slice_size = 100
    i = 0
    while i < len(img_paths):
        results = model(
            source=img_paths[i:i+slice_size],
            stream=True,
            conf=0.9
        )
        for result in results:
            if result.boxes:
                result.save(os.path.join(OUTPUT_DIR, 'yolo_' + result.path.split('/')[-1]))
        i += slice_size
    exit()

if args.forward:
    results = model(
        source=args.forward,
        stream=True
    )
    for i, result in enumerate(results):
        if result.boxes:
            result.save(os.path.join(OUTPUT_DIR, 'yolo_' + str(i) + '.jpg'))
    exit()