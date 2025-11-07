from ultralytics import YOLO

class YOLOModel:

    def __init__(self, model_path, device, conf_thres):
        self.device = device
        self.conf_thres = conf_thres
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