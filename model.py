import yaml
from argparse import ArgumentParser, BooleanOptionalAction
from yolo import YOLOModel

# arguments from the command line
parser = ArgumentParser(description='Bustine Detection YOLOv11')
parser.add_argument('--data', type=str, help='data.yaml path (can be set also in the congif.yaml file)')
parser.add_argument('--output', type=str, help='output directory for the model')
parser.add_argument('--model', type=str, help='model path')
parser.add_argument('--train', type=bool, help='start the training on the dataset', action=BooleanOptionalAction)
parser.add_argument('--test', type=bool, help='start the testing on the dataset', action=BooleanOptionalAction)
parser.add_argument('--forward', type=str, help='evaluate the model on a video')
parser.add_argument('--show_dataset', type=bool, help='show the dataset', action=BooleanOptionalAction)
parser.add_argument('--plot_result', type=str, help='plot the results from the results.csv file')
args = parser.parse_args()

# arguments from the config file
CONFIGS = yaml.safe_load(open('config.yaml'))

# extract yolo configuration
model_config = CONFIGS.get('model', {})

# merge conflicting arguments
DATA_FILE = args.data if args.data else model_config.get('data', {}).get('yaml_file', "./datasets/data.yaml")
OUTPUT_DIR = args.output if args.output else model_config.get('data', {}).get('output_folder', "./yolo_output")
MODEL = args.model if args.model else model_config.get('model_path', 'best.pt')
NUM_EPOCHS = model_config.get('train', {}).get('epochs', 100)
IMG_SIZE = model_config.get('train', {}).get('img_size') or 640
DEVICE = model_config.get('device', 'cuda:0')
CONF_THRES = model_config.get('confidence_threshold', 0.9)

yolo_model = YOLOModel(
    model_path=MODEL,
    device=DEVICE,
    conf_thres=CONF_THRES,
    data_file=DATA_FILE,
    output_dir=OUTPUT_DIR
)

if args.show_dataset:
    yolo_model.show_dataset()
    exit()

if args.plot_result:
    yolo_model.plot_result(args.plot_result)
    exit()

if args.train:
    yolo_model.train(epochs=NUM_EPOCHS, imgsz=IMG_SIZE)
    exit()

if args.test:
    yolo_model.test()
    exit()

if args.forward:
    yolo_model.evaluate_forward(args.forward)
    exit()