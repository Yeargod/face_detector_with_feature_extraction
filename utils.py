import cv2

def resize_image(img, yolo_model):
    [im_h, im_w, _] = img.shape
    [net_h, net_w] = [yolo_model.neth, yolo_model.netw]

    if(float(net_w)/im_w < float(net_h)/im_h): 
        new_w = net_w;
        scale_ratio = float(net_w)/im_w
        new_h = int(im_h * scale_ratio);
    else:
        new_h = net_h;
        scale_ratio = float(net_h)/im_h
        new_w = int(im_w * scale_ratio);
        
    new_h = new_h+1 if new_h%2==1 else new_h
    new_w = new_w+1 if new_w%2==1 else new_w
    img = cv2.resize(img, (new_w, new_h))   
    return img, scale_ratio