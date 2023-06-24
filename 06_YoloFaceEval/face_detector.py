import time
import cv2
import numpy as np
import timeit
import onnxruntime
import math
from face import Face

class FaceDetector():
    def __init__(self,
                model_address="./model.onnx",
                det_size = 1280,
                det_thresh = 0.3,
                iou_thres = 0.5,
                gpu_load = False):
        
        self.model_address = model_address
        self.img_size = det_size
        self.iou_thres = iou_thres
        self.conf_thres = det_thresh
        self.imgsz = (det_size, det_size)
        self.device = 'cpu'
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_address, providers=providers)
        print(f'Applied providers: {self.model._providers}, with options: {self.model._provider_options}')

    def letterbox(self, img, new_shape=(640, 640), color=(255, 255, 0), auto=False, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def check_img_size(self, img_size, s=32):
        # Verify img_size is a multiple of stride s
        new_size = self.make_divisible(img_size, int(s))  # ceil gs-multiple
        if new_size != img_size:
            print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
        return new_size

    def make_divisible(self, x, divisor):
        # Returns x evenly divisible by divisor
        return math.ceil(x / divisor) * divisor

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x) # x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.min(box1[:, None, 2:], box2[:, 2:]) -
        np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)

    def my_nms(self, boxes, score, score_thresh, iou_thresh):  
        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]
        
        # Picked bounding indexes
        indexes = []
        
        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)
        
        # Sort by confidence score of bounding boxes
        order = np.argsort(score)
        
        # Iterate bounding boxes
        while len(order) > 0:
            # The index of largest confidence score
            index = order[-1]
            
            # Pick the indexes with largest confidence score
            indexes.append(index)
            
            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])
            
            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h
            
            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
            
            left = np.where(ratio < iou_thresh)
            order = order[left]
        return np.array(indexes) # np.array(picked_boxes).astype(np.float32), np.array(picked_score).astype(np.float32)

    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results
        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """

        # nc = prediction.shape[2] - 15  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # print(np.where(xc == True))

        # Settings
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [np.zeros((0, 16))] *  prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Compute conf
            x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            scores = x[:, 15:]  
            scores = np.squeeze(scores)
            x[:, 4] = scores

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            x[:, :4] = self.xywh2xyxy(x[:, :4])
            boxes = x[:, :4]

            i = self.my_nms(boxes, scores, self.conf_thres, self.iou_thres)

            output[xi] = x[i] if np.any(i) else []

            if (time.time() - t) > time_limit:
                break  # time limit exceeded
        return output

    def dynamic_resize(self, shape, stride=64):
        max_size = max(shape[0], shape[1])
        if max_size % stride != 0:
            max_size = (int(max_size / stride) + 1) * stride 
        return max_size

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        #clip_coords(coords, img0_shape)
        np.clip(coords[:, 0], 0, img0_shape[1])
        np.clip(coords[:, 1], 0, img0_shape[0])
        np.clip(coords[:, 2], 0, img0_shape[1])
        np.clip(coords[:, 3], 0, img0_shape[0])
        np.clip(coords[:, 4], 0, img0_shape[1])
        np.clip(coords[:, 5], 0, img0_shape[0])
        np.clip(coords[:, 6], 0, img0_shape[1])
        np.clip(coords[:, 7], 0, img0_shape[0])
        np.clip(coords[:, 8], 0, img0_shape[1])
        np.clip(coords[:, 9], 0, img0_shape[0])
        return coords

    def show_results(self, img, xywh, conf, landmarks, class_num):
        h,w,c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

        tf = max(tl - 1, 1)  # font thickness
        label = str(int(class_num)) + ': ' + str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        np.clip(boxes[:, 0], 0, img_shape[1])
        np.clip(boxes[:, 1], 0, img_shape[0])
        np.clip(boxes[:, 2], 0, img_shape[1])
        np.clip(boxes[:, 3], 0, img_shape[0])

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x) #x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y
        
    def detection_preprocess(self, img0):
        stride = 64 # int(self.model.stride.max())  # model stride
        imgsz = self.img_size
        if imgsz <= 0:                    # original size    
            imgsz = self.dynamic_resize(img0.shape)
        imgsz = self.check_img_size(imgsz, s=64)  # check img_size
        img = self.letterbox(img0, imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        return img
    
    def inference(self, img0, dynamic):
        """
        The function to inference input image from model.
  
        Parameters:
            img0 (numpy.ndarray): The input frame, must be a numpy ndarray in shape of (H, W, 3) and BGR format.
          
        Returns:
            faces: List of detected faces, each element is a Face object instance, Face object parameters are bbox(x1, y1, x2, y2), kps, det_score, Example: Face(bbox=bbox, kps=kps, det_score=det_score)
            inference_time: Only model inference time (second)
        """

        #Preprocess
        img = self.detection_preprocess(img0) 
        
        # Inference
        start_time = timeit.default_timer()
        pred = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: img})[0] 
        inference_time = timeit.default_timer() - start_time

        # Postprocess
        pred = self.non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0] # Apply NMS

        faces = []
        if len(pred):
            pred[:, :4] = self.scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
            pred[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
            for j in range(len(pred)): #.size()[0]):
                pred_j = np.reshape(pred[j, :4], (1, 4))
                pred_j = np.squeeze(pred_j)
                kps_j = np.reshape(pred[j, 5:15], (5, 2))
                conf_j = pred[j, 4] #.cpu().numpy()

                bbox = ([int(pred_j[0]), int(pred_j[1]), int(pred_j[2]), int(pred_j[3]), conf_j])
                kps = kps_j
                det_score = conf_j

                cropped_face = img0[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
                face = Face(bbox=bbox, kps=kps, score=det_score, cropped_face=cropped_face)
                faces.append(face)
        return faces, inference_time


if __name__ == '__main__':
    detection_model = FaceDetector()
    source = "./frame.jpg"
    bgr_image = cv2.imread(source)
    faces, inference_time = detection_model.inference(bgr_image, dynamic=False)

    # Test
    bboxes = []
    confidences = []
    kps = []
    for face in faces:
        print(face)
        bboxes.append(face.bbox)
        confidences.append(face.score)
        kps.append(face.kps)

    print("bboxes = {}".format(bboxes))
    print("confidences = {}".format(confidences))
    print("inference_time = {}".format(inference_time))
    print("kpss={}".format(kps))

    for i, bbox in enumerate(bboxes):
        bgr_image = cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        for j in range(5):
            cv2.circle(bgr_image, center=(int(kps[i][j][0]), int(kps[i][j][1])), radius=1, color=(255, 255, 0))
    cv2.imwrite("onnx_img_numpy_last.jpg", bgr_image)
