import cv2
import cv2withpp
import time

def test3(self):
    x = (self['text'].cpt[0] + 1) % 600
    self['text'].update(cpt=(x, 100))

def click(self, selfobject, event, patam):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(patam)

def main():
    data = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 4]]
    layer = cv2withpp.Layer(width=800, height=600, color=(255, 255, 0))
    # bargraph = cv2withpp.BarGraph((200, 200), 300, 100, barw=15, baselinecolor=(0, 0, 0), data=data, color=[(0, 0, 255)])
    calender = cv2withpp.Calender((200, 200), 2023, 9, size=15, bgcolor=(255, 255, 255))
    calender.setEventLisnner(True)
    calender.addMouseEventCallback(click)
    layer['calender'] = calender
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