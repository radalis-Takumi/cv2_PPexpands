import cv2
import cv2withpp
import random

def main():
    img = cv2withpp.makeCanvas(800, 600, (255, 0, 255))
    layer = cv2withpp.Layer(img)
    figure = cv2withpp.Figure('test.png', (50, 100), rotate=45, mode='alpha')
    layer.append(figure)

    count = 3
    while True:
        show_img = layer.draw()
        cv2.imshow('window', show_img)
        count += 1
        key = cv2.waitKey(1000)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()