import cv2
import cv2withpp
import time

def test3(self):
    x = (self['text'].cpt[0] + 1) % 600
    self['text'].update(cpt=(x, 100))

def click(self, event):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('OK')

def main():
    layer = cv2withpp.Layer(width=800, height=600, color=(255, 255, 0))
    # figure = cv2withpp.Figure('./sample/test.png', (50, 100), rotate=45, mode='rgba')
    # rect = cv2withpp.Rectangle((50, 100), 50, 50, fillcolor=(0, 0, 255), rad=10)
    tri = cv2withpp.Triangle((50, 100), 80, 70, fillcolor=(0, 0, 255))
    # text = cv2withpp.Textbox('text', (50, 100), 'C:\Windows\Fonts\msgothic.ttc', 30, (255,0,0), framecolor=(0, 255, 0), anchor='la')
    tri.setEventLisnner(True)
    tri.addMouseEventCallback(click)
    layer['tri'] = tri
    # layer.addCallback_useSelf(test3)
    layer.displayMouse(True)
    layer.run_Async(interval=10)
    i = 0
    while not layer.getResult()[0]:
        time.sleep(1)
        print(i)
        i += 1

if __name__ == '__main__':
    main()