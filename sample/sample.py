import cv2
import cv2withpp

def main():
    img = cv2withpp.makeCanvas(800, 600, (255, 255, 255))
    rec1 = cv2withpp.Rectangle((50, 50), 50, 50, rad=10, rotate=0, fillcolor=(255, 0, 0), framecolor=(0, 0, 255))
    rec2 = cv2withpp.Rectangle((100, 100), 100, 100, rad=0, rotate=20, fillcolor=(255, 0, 0), framecolor=(0, 0, 255))
    text1 = cv2withpp.Textbox('textテキスト', (150, 150), 'C:\Windows\Fonts\msgothic.ttc', 30, (255,0,0), anchor='la')
    ellipse = cv2withpp.Ellipse((150, 150), (50, 50), 0, startAngle=90, fillcolor=(0, 255, 0))
    triangle = cv2withpp.Triangle((400, 400), 60, 60, fillcolor=(0, 255, 0))
    layer = cv2withpp.Layer(img)
    layer.append(rec2)
    layer.append(rec1)
    layer.append(ellipse)
    layer.append(text1)
    layer.append(triangle)
    count = 0
    while True:
        show_img = layer.draw()
        cv2.imshow('window', show_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()