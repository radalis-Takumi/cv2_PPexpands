import cv2
import cv2withpp
import time

def test3(self):
    x = (self['figure1'].cpt[0] + 1) % 600
    self['figure1'].update(cpt=(x, 100))

def main():
    layer = cv2withpp.Layer(width=800, height=600, color=(255, 0, 255))
    figure = cv2withpp.Figure('./sample/test.png', (50, 100), rotate=45, mode='alpha')
    layer['figure1'] = figure
    layer.addCallback_useSelf(test3)
    layer.run_Async(interval=100)
    for i in range(10):
        time.sleep(1)
        print(i)
        print(layer.getResult())

if __name__ == '__main__':
    main()