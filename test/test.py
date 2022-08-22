from authmmcls import AuthModel
import cv2


def main():
    image = cv2.imread('../../imgs/fraud/ft_bgr_3812609_0.png')
    # model = AuthModel(config_path='../configs/mobilenetv3.py', weights_path='../../epoch_18.pth')
    model = AuthModel(config_path='../configs/mobilenetv2.py', weights_path='../../best_accuracy_top-1_epoch_33.pth')
    pred = model.predict(image)
    print(pred)

if __name__ == '__main__':
    main()