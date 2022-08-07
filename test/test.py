from authmmcls import AuthModel
import cv2


def main():
    image = cv2.imread('../../imgs/fraud/ft_bgr_3812609_0.png')
    model = AuthModel('../../epoch_18.pth')
    pred = model.predict(image)
    print(pred)

if __name__ == '__main__':
    main()