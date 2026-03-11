import yaml
import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

class YOLOModel:

    def __init__(self, model_path, device, conf_thres=0.9, data_file=None, output_dir=None):
        self.device = device
        self.conf_thres = conf_thres
        self.data_file = data_file
        self.output_dir = output_dir
        self.model = YOLO(
            model=model_path,
            task='detect',
        )
    
    def forward(self, image):
        results = self.model(
            source=image,
            stream=True,
            verbose=False,
            device=self.device,
            conf=self.conf_thres,
        )
        for result in results:
            if result.boxes:
                return [b + [c] for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())]
        return None

    def show_dataset(self):
        if not self.data_file or not self.output_dir:
            print("show_dataset logic requires data_file and output_dir.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        DATA_CONFIGS = yaml.safe_load(open(self.data_file))
        dataset_root_dir = self.data_file.split('data.yaml')[0]
        image_dir = os.path.join(dataset_root_dir, 'images')
        label_dir = os.path.join(dataset_root_dir, 'labels')

        image_output_dir = os.path.join(self.output_dir, 'dataset')
        os.makedirs(image_output_dir, exist_ok=True)

        for subset in ['train', 'val', 'test']:
            image_subset_dir = os.path.join(image_dir, subset)
            label_subset_dir = os.path.join(label_dir, subset)

            if not os.path.exists(image_subset_dir):
                continue

            for image in os.listdir(image_subset_dir):
                image_path = os.path.join(image_subset_dir, image)
                label_path = os.path.join(label_subset_dir, image.replace('jpg', 'txt'))
                
                # read the image and the label
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img_height, img_width, _ = img.shape

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            _, x_center, y_center, bbox_width, bbox_height = [float(x) for x in line.split(' ')]
                        
                            x1 = int((x_center - bbox_width / 2) * img_width)
                            y1 = int((y_center - bbox_height / 2) * img_height)
                            x2 = int((x_center + bbox_width / 2) * img_width)
                            y2 = int((y_center + bbox_height / 2) * img_height)
                            
                            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                cv2.imwrite(os.path.join(image_output_dir, image), img)

    def plot_result(self, file_path):
        print('Plotting the results')
        plot_results(file=file_path)

    def train(self, epochs=100, imgsz=640, additional_args=None):
        if not self.data_file:
            print("No data file specified for training.")
            return
        
        train_args = {
            "data": self.data_file,
            "epochs": epochs,
            "device": self.device,
            "imgsz": imgsz,
            "dropout": 0.8,
            "degrees": 0,
            "translate": 0,
            "scale": 0,
            "shear": 0,
            "perspective": 0,
            "bgr": 0,
            "mosaic": 0,
            "mixup": 0,
            "copy_paste": 0,
            "erasing": 0,
        }
        if additional_args:
            train_args.update(additional_args)

        train_results = self.model.train(**train_args)
        return train_results

    def test(self):
        if not self.data_file:
            print("No data file specified for testing.")
            return
        eval_results = self.model.val(data=self.data_file, device=self.device)
        return eval_results

    def evaluate_forward(self, forward_arg):
        if not self.output_dir:
            print("Output dir required.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        if forward_arg in ['test', 'train', 'val']:
            if not self.data_file:
                print("No data file specified.")
                return
            data_path = yaml.safe_load(open(self.data_file))
            if forward_arg not in data_path:
                print(f"Split {forward_arg} not found in data file.")
                return

            base_img_dir = os.path.join('./datasets/images/', forward_arg)
            img_paths = [os.path.join(base_img_dir, img) for img in os.listdir(os.path.join('./datasets', data_path[forward_arg]))]
            slice_size = 100
            i = 0
            while i < len(img_paths):
                results = self.model(
                    source=img_paths[i:i+slice_size],
                    stream=True,
                    conf=self.conf_thres,
                    device=self.device
                )
                for result in results:
                    if result.boxes:
                        result.save(os.path.join(self.output_dir, 'yolo_' + result.path.split('/')[-1]))
                i += slice_size
        else:
            results = self.model(
                source=forward_arg,
                stream=True,
                conf=self.conf_thres,
                device=self.device
            )
            for i, result in enumerate(results):
                if result.boxes:
                    result.save(os.path.join(self.output_dir, 'yolo_' + str(i) + '.jpg'))