import cv2
import cv2withpp
import random

def test3(arg1, arg2, arg3):
    if arg1:
        print(arg2)
    else:
        print(arg3)

def main():
    layer = cv2withpp.Layer(width=800, height=600, color=(255, 0, 255))
    figure = cv2withpp.Figure('./sample/test.png', (50, 100), rotate=45, mode='alpha')
    layer.append(figure)
    layer.add_func(test3, arg=(False, 'OK', 'NG'))
    layer.add_func(test3, arg=(True, 'OK', 'NG'))
    layer.run()

if __name__ == '__main__':
    main()