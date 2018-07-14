./darknet detector train cfg/face-deep-size448-anchor10.data cfg/yolo-face-deep-size448-anchor10.cfg darknet19_448.conv.23 -gpus 0,1
./darknet detector train cfg/face-deep-size608-anchor10.data cfg/yolo-face-deep-size608-anchor10.cfg darknet19_448.conv.23 -gpus 0,1
./darknet detector train cfg/face-deep-size608-anchor20.data cfg/yolo-face-deep-size608-anchor20.cfg darknet19_448.conv.23 -gpus 0,1
./darknet detector train cfg/face-shallow-size608-anchor10.data cfg/yolo-face-shallow-size608-anchor10.cfg darknet19_448.conv.23 -gpus 0,1
./darknet detector train cfg/face-shallow-size608-anchor20.data cfg/yolo-face-shallow-size608-anchor20.cfg darknet19_448.conv.23 -gpus 0,1
