import cv2
import numpy as np
import calendar
import jpholiday
import datetime
from PIL import Image, ImageDraw, ImageFont
import math

def makeCanvas(width, height, color=(255, 255, 255)):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, -1)
    return img

def cv2_putText(img, text, org, fontFace, fontScale, color, mode=None, anchor=None):
    """
    mode:
        0:left bottom, 1:left ascender, 2:middle middle,
        3:left top, 4:left baseline
    anchor:
        lb:left bottom, la:left ascender, mm: middle middle,
        lt:left top, ls:left baseline
    """

    # テキスト描画域を取得
    x, y = org
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    dummy_draw = ImageDraw.Draw(Image.new("L", (0,0)))
    xL, yT, xR, yB = dummy_draw.multiline_textbbox((x, y), text, font=fontPIL)

    # modeおよびanchorによる座標の変換
    img_h, img_w = img.shape[:2]
    if mode is None and anchor is None:
        offset_x, offset_y = xL - x, yB - y
    elif mode == 0 or anchor == "lb":
        offset_x, offset_y = xL - x, yB - y
    elif mode == 1 or anchor == "la":
        offset_x, offset_y = 0, 0
    elif mode == 2 or anchor == "mm":
        offset_x, offset_y = (xR + xL)//2 - x, (yB + yT)//2 - y
    elif mode == 3 or anchor == "lt":
        offset_x, offset_y = xL - x, yT - y
    elif mode == 4 or anchor == "ls":
        _, descent = ImageFont.FreeTypeFont(fontFace, fontScale).getmetrics()
        offset_x, offset_y = xL - x, yB - y - descent

    x0, y0 = x - offset_x, y - offset_y
    xL, yT = xL - offset_x, yT - offset_y
    xR, yB = xR - offset_x, yB - offset_y

    # バウンディングボックスを描画　不要ならコメントアウトする
    cv2.rectangle(img, (xL,yT), (xR,yB), color, 1)

    # 画面外なら何もしない
    if xR<=0 or xL>=img_w or yB<=0 or yT>=img_h:
        print("out of bounds")
        return img

    # ROIを取得する
    x1, y1 = max([xL, 0]), max([yT, 0])
    x2, y2 = min([xR, img_w]), min([yB, img_h])
    roi = img[y1:y2, x1:x2]

    # ROIをPIL化してテキスト描画しCV2に戻る
    roiPIL = Image.fromarray(roi)
    draw = ImageDraw.Draw(roiPIL)
    draw.text((x0-x1, y0-y1), text, color, fontPIL)
    roi = np.array(roiPIL, dtype=np.uint8)
    img[y1:y2, x1:x2] = roi

    return img

class Textbox:
    def __init__(self, text, org, fontFace, fontScale=10, fontcolor=(0,0,0), framecolor=None, mode=0, anchor=None):
        """
        mode:
            0:left bottom, 1:left ascender, 2:middle middle,
            3:left top, 4:left baseline
        anchor:
            lb:left bottom, la:left ascender, mm: middle middle,
            lt:left top, ls:left baseline
        """
        self.text = text
        self.cpt = org
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.fontcolor = fontcolor
        self.framecolor = framecolor
        self.mode = mode
        if anchor and anchor in ['lb', 'la', 'mm', 'lt', 'ls']:
            self.mode = {'lb':0, 'la':1, 'mm':2, 'lt':3, 'ls':4}[anchor]
        self.setAria()
        
    # テキスト描画域を取得
    def setAria(self):
        self.fontPIL = ImageFont.truetype(font = self.fontFace, size = self.fontScale)
        dummy_draw = ImageDraw.Draw(Image.new("L", (0,0)))
        xL, yT, xR, yB = dummy_draw.multiline_textbbox(self.cpt, self.text, font=self.fontPIL)

        # modeおよびanchorによる座標の変換
        if self.mode == 0:
            offset_x, offset_y = xL - self.cpt[0], yB - self.cpt[1]
        elif self.mode == 1:
            offset_x, offset_y = 0, 0
        elif self.mode == 2:
            offset_x, offset_y = (xR + xL)//2 - self.cpt[0], (yB + yT)//2 - self.cpt[1]
        elif self.mode == 3:
            offset_x, offset_y = xL - self.cpt[0], yT - self.cpt[1]
        else: ## self.mode == 4:
            _, descent = ImageFont.FreeTypeFont(self.fontFace, self.fontScale).getmetrics()
            offset_x, offset_y = xL - self.cpt[0], yB - self.cpt[1] - descent

        x0, y0 = self.cpt[0] - offset_x, self.cpt[1] - offset_y
        xL, yT = xL - offset_x, yT - offset_y
        xR, yB = xR - offset_x, yB - offset_y
        self.Ariaorg = {'x0':x0, 'y0':y0, 'xL':xL, 'yT':yT, 'xR':xR, 'yB':yB}
    
    def setPram(self, fontFace=None, fontScale=None, mode=None):
        if fontFace:
            self.fontFace = fontFace
        if fontScale:
            self.fontScale = fontScale
        if mode:
            self.mode = mode
        self.setAria()
    
    def draw(self, img):
        img_h, img_w = img.shape[:2]
        # 枠線を描画
        if self.framecolor:
            cv2.rectangle(img, (self.Ariaorg['xL'], self.Ariaorg['yT']),
                           (self.Ariaorg['xR'], self.Ariaorg['yB']), self.framecolor, 1)

        # 画面外なら何もしない
        if self.Ariaorg['xR']<=0 or self.Ariaorg['xL']>=img_w or self.Ariaorg['yB']<=0 or self.Ariaorg['yT']>=img_h:
            print("out of bounds")
            return img

        # ROIを取得する
        x1, y1 = max([self.Ariaorg['xL'], 0]), max([self.Ariaorg['yT'], 0])
        x2, y2 = min([self.Ariaorg['xR'], img_w]), min([self.Ariaorg['yB'], img_h])
        roi = img[y1:y2, x1:x2]

        # ROIをPIL化してテキスト描画しCV2に戻る
        roiPIL = Image.fromarray(roi)
        draw = ImageDraw.Draw(roiPIL)
        draw.text((self.Ariaorg['x0']-x1, self.Ariaorg['y0']-y1), self.text, self.fontcolor, self.fontPIL)
        roi = np.array(roiPIL, dtype=np.uint8)
        img[y1:y2, x1:x2] = roi

        return img

class Triangle:
    def __init__(self, cpt, width, height, rotate=0, fillcolor=None, framecolor=None):
        self.cpt = cpt
        self.width = width
        self.height = height
        self.rotate = rotate
        self.fillcolor = fillcolor
        self.framecolor = framecolor
        self.para_change()
    
    def obj_move(self, cpt):
        self.pts = []
        for pt in self.b_pts:
            self.pts.append([round(cpt[0] + pt[0]), round(cpt[1] + pt[1])])
    
    def obj_rotete(self, rotate):
        self.b_pts = [[            0, -self.height * 2/3],
                      [ self.width/2,  self.height/3], 
                      [-self.width/2,  self.height/3]]
        radians = -math.radians(rotate) # 回転方向を一般的な感覚（反時計回り）に変更
        for i, pt in enumerate(self.b_pts):
            self.b_pts[i] = [math.cos(radians)*pt[0] - math.sin(radians)*pt[1], math.sin(radians)*pt[0] + math.cos(radians)*pt[1]]
    
    def para_change(self, rotate=None, cpt=None):
        if rotate:
            self.rotate = rotate % 360
        if cpt:
            self.cpt = cpt
        self.obj_rotete(self.rotate)
        self.obj_move(self.cpt)

    def draw(self, img):
        pts = np.array(self.pts)
        if self.fillcolor:
            cv2.fillPoly(img, [pts], self.fillcolor, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.polylines(img, [pts], True, self.framecolor, thickness=1, lineType=cv2.LINE_AA)
        return img

class Rectangle:
    def __init__(self, cpt, width, height, rad=0, rotate=0, fillcolor=None, framecolor=None):
        self.cpt = cpt
        self.width = width
        self.height = height
        self.rad = rad if rad <= min(self.width/2, self.height) else min(self.width/2, self.height)
        self.rotate = rotate % 360
        self.fillcolor = fillcolor
        self.framecolor = framecolor
        self.para_change()
        
    def obj_move(self, cpt):
        self.pts1 = []
        self.pts2 = []
        self.radc_pts = []
        for pt in self.b_pts1:
            self.pts1.append([round(cpt[0] + pt[0]), round(cpt[1] + pt[1])])
        for pt in self.b_pts2:
            self.pts2.append([round(cpt[0] + pt[0]), round(cpt[1] + pt[1])])
        for pt in self.radcb_pts:
            self.radc_pts.append([round(cpt[0] + pt[0]), round(cpt[1] + pt[1])])
    
    def obj_rotete(self, rotate):
        radians = -math.radians(rotate) # 回転方向を一般的な感覚（反時計回り）に変更
        for i, pt in enumerate(self.b_pts1):
            self.b_pts1[i] = [math.cos(radians)*pt[0] - math.sin(radians)*pt[1], math.sin(radians)*pt[0] + math.cos(radians)*pt[1]]
        for i, pt in enumerate(self.b_pts2):
            self.b_pts2[i] = [math.cos(radians)*pt[0] - math.sin(radians)*pt[1], math.sin(radians)*pt[0] + math.cos(radians)*pt[1]]
        for i, pt in enumerate(self.radcb_pts):
            self.radcb_pts[i] = [math.cos(radians)*pt[0] - math.sin(radians)*pt[1], math.sin(radians)*pt[0] + math.cos(radians)*pt[1]]

    def obj_radChange(self, rad):
        self.b_pts1 = [[-(self.width/2 - rad), -self.height/2],
                       [ (self.width/2 - rad), -self.height/2], 
                       [ (self.width/2 - rad),  self.height/2], 
                       [-(self.width/2 - rad),  self.height/2]]
        self.b_pts2 = [[-self.width/2, -(self.height/2 - rad)],
                       [ self.width/2, -(self.height/2 - rad)], 
                       [ self.width/2,  (self.height/2 - rad)], 
                       [-self.width/2,  (self.height/2 - rad)]]
        self.radcb_pts = [[-(self.width/2 - rad), -(self.height/2 - rad)],
                         [ (self.width/2 - rad), -(self.height/2 - rad)], 
                         [ (self.width/2 - rad),  (self.height/2 - rad)], 
                         [-(self.width/2 - rad),  (self.height/2 - rad)]]
    
    def para_change(self, rad=None, rotate=None, cpt=None):
        if not rad is None:
            self.rad = rad if rad <= min(self.width/2, self.height) else min(self.width/2, self.height)
        if rotate:
            self.rotate = rotate % 360
        if cpt:
            self.cpt = cpt
        self.obj_radChange(self.rad)
        self.obj_rotete(self.rotate)
        self.obj_move(self.cpt)

    def draw(self, img):
        h, w, _ = img.shape
        tmpimg = np.full((h, w, 1), 0, dtype=np.uint8)
        pts1 = np.array(self.pts1)
        pts2 = np.array(self.pts2)
        cv2.fillPoly(tmpimg, [pts1], (255, 255, 255), lineType=cv2.LINE_AA, shift=0)
        cv2.fillPoly(tmpimg, [pts2], (255, 255, 255), lineType=cv2.LINE_AA, shift=0)
        for c_pts in self.radc_pts:
            cv2.circle(tmpimg, (c_pts[0], c_pts[1]), self.rad, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        contours, hierarchy = cv2.findContours(tmpimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.fillcolor:
            cv2.drawContours(img, contours, -1, self.fillcolor, -1, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.drawContours(img, contours, -1, self.framecolor, 1, lineType=cv2.LINE_AA)
        return img
    
class Ellipse:
    def __init__(self, cpt, axes, rotate, startAngle=0, endAngle=360, fillcolor=None, framecolor=None):
        self.cpt = cpt
        self.axes = axes
        self.rotate = rotate
        self.startAngle = startAngle
        self.endAngle = endAngle
        self.fillcolor = fillcolor
        self.framecolor = framecolor

    def draw(self, img):
        if self.fillcolor:
            cv2.ellipse(img, self.cpt, self.axes, -self.rotate, -self.startAngle, -self.endAngle, self.fillcolor, thickness=-1, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.ellipse(img, self.cpt, self.axes, -self.rotate, -self.startAngle, -self.endAngle, self.framecolor, thickness=1, lineType=cv2.LINE_AA)

class Calender:
    def __init__(self, cpt, year, month, size=10):
        self.year = year
        self.month = month
        self.cpt = cpt
        self.size = size
        self.ymUpdate()
    
    def ymUpdate(self, cpt=None, year=None, month=None):
        if cpt:
            self.cpt = cpt
        if not year is None:
            self.year = year
        if not month is None:
            self.month = month
        self.txt_list = []
        textm = f'{self.month}' if self.month > 9 else f' {self.month}'
        self.txt_list.append(Textbox(f'{self.year}年  {textm}月', (self.cpt[0], self.cpt[1] - self.size * 7), 
                                     'C:\Windows\Fonts\msgothic.ttc', self.size, (0,0,0), anchor='mm'))
        for i, day in enumerate(['月', '火', '水', '木', '金', '土', '日']):
            self.txt_list.append(Textbox(day, (self.cpt[0] - self.size * (6 - i * 2), self.cpt[1] - self.size * 5), 
                                     'C:\Windows\Fonts\msgothic.ttc', self.size, (0,0,0), anchor='mm'))
        for i, week_list in enumerate(calendar.monthcalendar(self.year, self.month)):
            for j, day in enumerate(week_list):
                if day == 0:
                    continue
                self.txt_list.append(Textbox(str(day), (self.cpt[0] - self.size * (6 - j * 2), self.cpt[1] - self.size * (3 - i * 2)), 
                                     'C:\Windows\Fonts\msgothic.ttc', self.size, (0,0,0), anchor='mm'))

    def draw(self, img):
        cv2.rectangle(img, (round(self.cpt[0] - self.size * 7.3), round(self.cpt[1] - self.size * 8.3)), 
                      (round(self.cpt[0] + self.size * 7.3), round(self.cpt[1] + self.size * 8.3)), (0, 0, 0), thickness=1)
        cv2.rectangle(img, (round(self.cpt[0] - self.size * 7), round(self.cpt[1] - self.size * 8)), 
                      (round(self.cpt[0] + self.size * 7), round(self.cpt[1] + self.size * 8)), (0, 0, 0), thickness=1)
        cv2.rectangle(img, (self.cpt[0] - self.size * 4, self.cpt[1] - self.size * 8), (self.cpt[0] + self.size * 4, self.cpt[1] - self.size * 6),
                       (0, 0, 0), thickness=1)
        for i, week in enumerate([[0, 0, 0, 0, 0, 0, 0]] + calendar.monthcalendar(self.year, self.month)):
            for j, day in enumerate(week):
                if j == 5:
                    cv2.rectangle(img, (self.cpt[0] - self.size * (7 - j * 2), self.cpt[1] - self.size * (6 - i * 2)),
                       (self.cpt[0] - self.size * (5 - j * 2), self.cpt[1] - self.size * (4 - i * 2)), (255, 144, 30), thickness=-1)
                elif j == 6 or (day > 0 and jpholiday.is_holiday(datetime.date(self.year, self.month, day))):  
                    cv2.rectangle(img, (self.cpt[0] - self.size * (7 - j * 2), self.cpt[1] - self.size * (6 - i * 2)),
                       (self.cpt[0] - self.size * (5 - j * 2), self.cpt[1] - self.size * (4 - i * 2)), (0, 0, 255), thickness=-1)
                cv2.rectangle(img, (self.cpt[0] - self.size * (7 - j * 2), self.cpt[1] - self.size * (6 - i * 2)),
                       (self.cpt[0] - self.size * (5 - j * 2), self.cpt[1] - self.size * (4 - i * 2)), (0, 0, 0), thickness=1)
        for txtobj in self.txt_list:
            txtobj.draw(img)
        return img


class Layer:
    def __init__(self, img):
        self.objectList = []
        self.base_img = img.copy()

    def update_baseimg(self, img):
        self.base_img = img.copy()
    
    def append(self, obj):
        self.objectList.append(obj)

    def draw(self, img=None):
        if img:
            draw_img = img.copy()
        else:
            draw_img = self.base_img.copy()
        for obj in self.objectList:
            obj.draw(draw_img)
        return draw_img