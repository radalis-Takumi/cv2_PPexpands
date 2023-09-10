import cv2
import numpy as np
import calendar
import jpholiday
import datetime
from PIL import Image, ImageDraw, ImageFont
import math
import queue
from concurrent import futures

def makeCanvas(width, height, color=(255, 255, 255)):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, -1)
    return img

class cv2withPPObject:
    def __init__(self, kay):
        self.kay = kay
        self.display = True
        self.eventLisnner = False
        self.callback = []
        self.mouseEventCallback = []
    
    def getEventLisnner(self):
        return self.eventLisnner
    
    def setEventLisnner(self, flag: bool):
        self.eventLisnner = flag
        
    def setKey(self, key):
        self.kay = key
    
    def getKey(self):
        return self.kay

    def setDisplay(self, flag):
        self.display = flag

    def getDisplay(self):
        return self.display
    
    def addAnimationCallback(self, callback):
        self.callback.append(callback)

    def addMouseEventCallback(self, callback):
        self.mouseEventCallback.append(callback)

    def isInner(self, base_vec, point):
        # 0行目 -> 末行にした行列を作る
        dim_vec = base_vec[list(range(1, base_vec.shape[0])) + [0], :]

        # ターゲットの座標を連続した行列を作る
        target_vec = np.tile(np.array(point), (base_vec.shape[0], 1))
        
        # 外積を求める
        cross_vec = np.cross(base_vec - dim_vec, target_vec - dim_vec)

        # 内外の判定を行う
        return np.all(cross_vec >= 0)


class Textbox(cv2withPPObject):
    """テキストボックス生成クラス
    Attributes:
        text (str): 表示する文字列
        cpt (tuple): 文字列を表示する座標
        fontFace (str): フォントファイルのパス
        fontScale (int): フォントサイズ
        fontcolor (tuple): フォントカラー
        framecolor (tuple): 枠線の色（引数を指定しなかった場合枠線なし）
        mode (int): 文字列を表示する座標の基準
    """
    def __init__(self, text:str, org:tuple, fontFace:str, fontScale=10, fontcolor=(0,0,0), framecolor=None, mode=0, anchor=None, key=''):
        """
        text:
            表示する文字列
        org:
            文字列を表示する座標
        fontFace:
            フォントファイルのパス
        fontScale:
            フォントサイズ
        fontcolor:
            フォントカラー
        framecolor:
            枠線の色（引数を指定しなかった場合枠線なし）
        mode:
            文字列を表示する座標の基準
            0:left bottom, 1:left ascender, 2:middle middle,
            3:left top, 4:left baseline
        anchor:
            lb:left bottom, la:left ascender, mm: middle middle,
            lt:left top, ls:left baseline
        """
        super().__init__(key)
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
    
    def update(self, text=None, fontFace=None, fontScale=None, mode=None):
        if not text is None:
            self.text = text
        if fontFace:
            self.fontFace = fontFace
        if fontScale:
            self.fontScale = fontScale
        if mode:
            self.mode = mode
        self.setAria()

    def isPtsInner(self, mouse_pts):
        pts = np.array([[self.Ariaorg['xL'], self.Ariaorg['yT']],
                        [self.Ariaorg['xL'], self.Ariaorg['yB']],
                        [self.Ariaorg['xR'], self.Ariaorg['yB']],
                        [self.Ariaorg['xR'], self.Ariaorg['yT']]])
        return self.isInner(pts, mouse_pts)
    
    def draw(self, img):
        for callback in self.callback:
            callback()
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

class Triangle(cv2withPPObject):
    """三角形生成クラス
    Attributes:
        cpt (tuple): 三角形の中心座標
        width (int): 三角形の底辺の長さ
        height (int): 三角形の高さ
        rotate (int): 回転角度(degree)（反時計回り）
        fillcolor (tuple): 塗り潰し色（引数を指定しなかった場合枠線なし）
        framecolor (tuple): 枠線の色（引数を指定しなかった場合枠線なし）
    """
    def __init__(self, cpt, width, height, rotate=0, fillcolor=None, framecolor=None, key=''):
        super().__init__(key)
        self.cpt = cpt
        self.width = width
        self.height = height
        self.rotate = rotate
        self.fillcolor = fillcolor
        self.framecolor = framecolor
        self.update()
        
    def obj_move(self, cpt):
        self.pts = []
        for pt in self.b_pts:
            self.pts.append([round(cpt[0] + pt[0]), round(cpt[1] + pt[1])])
    
    def obj_rotete(self, rotate):
        self.b_pts = [[            0, -self.height * 2/3],
                      [-self.width/2,  self.height/3], 
                      [ self.width/2,  self.height/3]]
        radians = -math.radians(rotate) # 回転方向を一般的な感覚（反時計回り）に変更
        for i, pt in enumerate(self.b_pts):
            self.b_pts[i] = [math.cos(radians)*pt[0] - math.sin(radians)*pt[1], math.sin(radians)*pt[0] + math.cos(radians)*pt[1]]
    
    def update(self, rotate=None, cpt=None):
        if rotate:
            self.rotate = rotate % 360
        if cpt:
            self.cpt = cpt
        self.obj_rotete(self.rotate)
        self.obj_move(self.cpt)

    def isPtsInner(self, mouse_pts):
        return self.isInner(np.array(self.pts), mouse_pts)

    def draw(self, img):
        for callback in self.callback:
            callback()
        pts = np.array(self.pts)
        if self.fillcolor:
            cv2.fillPoly(img, [pts], self.fillcolor, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.polylines(img, [pts], True, self.framecolor, thickness=1, lineType=cv2.LINE_AA)
        return img

class Rectangle(cv2withPPObject):
    """
    四角形生成クラス
    -------------
    Parameters

    cpt (tuple):
        四角形の中心座標
    width (int):
        四角形の横の長さ
    height (int):
        四角形の縦の長さ
    rad (int):
        四隅の半径
    rotate (int):
        回転角度(degree)（反時計回り）
    fillcolor (tuple):
        塗り潰し色（引数を指定しなかった場合枠線なし）
    framecolor (tuple):
        枠線の色（引数を指定しなかった場合枠線なし）
    """
    def __init__(self, cpt, width, height, rad=0, rotate=0, fillcolor=None, framecolor=None, key=''):
        super().__init__(key)
        self.cpt = cpt
        self.width = width
        self.height = height
        self.rad = rad if rad <= min(self.width/2, self.height) else min(self.width/2, self.height)
        self.rotate = rotate % 360
        self.fillcolor = fillcolor
        self.framecolor = framecolor
        self.update(rad=self.rad)
    
    def set_contours(self):
        tmpimg = np.full((self.height + 10, self.width + 10, 1), 0, dtype=np.uint8)
        b_pts = [[round(x + self.width/2 + 5), round(y + self.height/2 + 5)] for x, y in self.b_pts]
        radcb_pts = [[round(x + self.width/2 + 5), round(y + self.height/2 + 5)] for x, y in self.radcb_pts]
        pts = np.array(b_pts)
        cv2.fillPoly(tmpimg, [pts], (255, 255, 255), lineType=cv2.LINE_AA, shift=0)
        for c_pts in radcb_pts:
            cv2.circle(tmpimg, tuple(c_pts), self.rad, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        contours, hierarchy = cv2.findContours(tmpimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.b_contours = contours[0] - np.array([round(self.width/2 + 5), round(self.height/2 + 5)])

    def obj_move(self, cpt):
        self.contours = self.contours + np.array(cpt)
    
    def obj_rotete(self, rotate):
        radians = -math.radians(rotate) # 回転方向を一般的な感覚（反時計回り）に変更
        self.contours = np.dot(self.b_contours, np.array([[math.cos(radians), math.sin(radians)], [-math.sin(radians), math.cos(radians)]]))
        self.contours = np.round(self.contours).astype(int)

    def obj_radChange(self, rad):
        self.b_pts = [[-(self.width/2 - rad), -self.height/2],
                      [ (self.width/2 - rad), -self.height/2], 
                      [ self.width/2, -(self.height/2 - rad)], 
                      [ self.width/2,  (self.height/2 - rad)], 
                      [ (self.width/2 - rad),  self.height/2], 
                      [-(self.width/2 - rad),  self.height/2], 
                      [-self.width/2,  (self.height/2 - rad)], 
                      [-self.width/2, -(self.height/2 - rad)]]
        self.radcb_pts = [[-(self.width/2 - rad), -(self.height/2 - rad)],
                         [ (self.width/2 - rad), -(self.height/2 - rad)], 
                         [ (self.width/2 - rad),  (self.height/2 - rad)], 
                         [-(self.width/2 - rad),  (self.height/2 - rad)]]
    
    def update(self, rad=None, rotate=None, cpt=None):
        if not rad is None:
            self.rad = rad if rad <= min(self.width/2, self.height) else min(self.width/2, self.height)
            self.obj_radChange(self.rad)
            self.set_contours()
        if rotate:
            self.rotate = rotate % 360
        if cpt:
            self.cpt = cpt
        self.obj_rotete(self.rotate)
        self.obj_move(self.cpt)

    def isPtsInner(self, mouse_pts):
        return self.isInner(self.contours[:, 0, :], mouse_pts)

    def draw(self, img):
        for callback in self.callback:
            callback()
        if self.fillcolor:
            cv2.drawContours(img, (self.contours,), -1, self.fillcolor, -1, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.drawContours(img, (self.contours,), -1, self.framecolor, 1, lineType=cv2.LINE_AA)
        return img
    
class Ellipse(cv2withPPObject):
    def __init__(self, cpt, axes, rotate, startAngle=0, endAngle=360, fillcolor=None, framecolor=None, key=''):
        super().__init__(key)
        self.cpt = cpt
        self.axes = axes
        self.rotate = rotate
        self.startAngle = startAngle
        self.endAngle = endAngle
        self.fillcolor = fillcolor
        self.framecolor = framecolor
    
    def isPtsInner(self, mouse_pts):
        x, y = mouse_pts
        x_d, y_d = self.cpt
        x_r, y_r = self.axes
        judge = True if ((x - x_d) / x_r)**2 + ((y - y_d) / y_r)**2 - 1 < 0 else False
        return judge

    
    def draw(self, img):
        for callback in self.callback:
            callback()
        if self.fillcolor:
            cv2.ellipse(img, self.cpt, self.axes, -self.rotate, -self.startAngle, -self.endAngle, self.fillcolor, thickness=-1, lineType=cv2.LINE_AA)
        if self.framecolor:
            cv2.ellipse(img, self.cpt, self.axes, -self.rotate, -self.startAngle, -self.endAngle, self.framecolor, thickness=1, lineType=cv2.LINE_AA)

class BarGraph(cv2withPPObject):
    def __init__(self, cpt, width, height, barw=2, baselinecolor=None, data=[], color=[(0,0,0)], key=''):
        super().__init__(key)
        self.cpt = cpt
        self.width = width
        self.height = height
        self.barw = barw
        self.baselinecolor = baselinecolor
        self.color = color
        self.display = True
        self.inputData(data)
    
    def update(self, data):
        self.data = data
        self.maxh = 1
        for v in self.data:
            for elv in v[1:]:
                if elv > self.maxh:
                    self.maxh = elv
    
    def setDisplay(self, flag):
        self.display = flag
        
    def getDisplay(self):
        return self.display
    
    def draw(self, img):
        for callback in self.callback:
            callback()
        if self.baselinecolor:
            cv2.line(img, (round(self.cpt[0] - self.width/2), self.cpt[1]), (round(self.cpt[0] + self.width/2), self.cpt[1]), self.baselinecolor, thickness=1)
        for i, v in enumerate(self.data):
            for j, w in enumerate(v[1:]):
                tmp_barw = self.barw/len(v[1:])
                x_pt = self.cpt[0] - (len(self.data) - 1 - i*2) * self.width/(len(self.data) * 2)
                y_pt = w * self.height/self.maxh
                cv2.rectangle(img, (round(x_pt - self.barw/2 + tmp_barw*j), round(self.cpt[1] - y_pt)), 
                            (round(x_pt - self.barw/2 + tmp_barw*(j+1)), round(self.cpt[1])), self.color[j], thickness=-1)
        return img

class Calender(cv2withPPObject):
    def __init__(self, cpt, year, month, size=10, key=''):
        super().__init__(key)
        self.year = year
        self.month = month
        self.cpt = cpt
        self.size = size
        self.update()
    
    def update(self, cpt=None, year=None, month=None):
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
        for callback in self.callback:
            callback()
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
    
class Figure(cv2withPPObject):
    def __init__(self, path, cpt, width=None, height=None, scale=1.0, w_scale=1.0, h_scale=1.0, rotate=0, mode='rgb', key=''):
        '''
        図を挿入するためのオブジェクト

        サイズ指定の優先順位
        width, height > scale > w_scale, h_scale
        '''
        super().__init__(key)
        self.figure = None
        self.path = path
        self.mode = mode
        self.width = width
        self.height = height
        self.scale = scale
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.cpt = cpt
        self.rotate = rotate
        self.update()
    
    def loadImgFile(self):
        if self.mode == 'rgba':
            self.figure = cv2.imread(self.path, flags=cv2.IMREAD_UNCHANGED)
        else: # mode == 'normal'
            self.figure = cv2.imread(self.path)
        if self.width and self.height:
            dsize = list(self.figure.shape[:2])
            if self.height:
                dsize[0] = self.height
            if self.width:
                dsize[1] = self.width
            self.figure = cv2.resize(self.figure, (dsize[1], dsize[0]))
        elif self.scale != 1.0 or self.w_scale != 1.0 or self.h_scale != 1.0:
            if self.scale != 1.0:
                self.figure = cv2.resize(self.figure, None, fx = self.scale, fy = self.scale)
            else:
                self.figure = cv2.resize(self.figure, None, fx = self.w_scale, fy = self.h_scale)
    
    def setupIMG(self):
        h, w = self.figure.shape[:2]
        radian = math.radians(self.rotate)
        # 回転後の画像サイズを計算
        w_rot = int(np.round(h*np.absolute(np.sin(radian))+w*np.absolute(np.cos(radian))))
        h_rot = int(np.round(h*np.absolute(np.cos(radian))+w*np.absolute(np.sin(radian))))
        size_rot = (w_rot, h_rot)

        # 元画像の中心を軸に回転する
        rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), self.rotate, 1.0)

        # 平行移動を加える (rotation + translation)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

        # 画像を実際に回転させる
        img_rot = cv2.warpAffine(self.figure, affine_matrix, size_rot, borderValue=(0, 255, 0))

        # 透過処理
        if self.mode == 'rgb':
            # 任意色部分に対応するマスク画像を生成
            mask = np.all(img_rot[:,:,:] == [0, 255, 0], axis=-1)
            # 元画像をBGR形式からBGRA形式に変換
            self.dst = cv2.cvtColor(img_rot, cv2.COLOR_BGR2BGRA)
            # マスク画像をもとに、白色部分を透明化
            self.dst[mask,3] = 0
        else:
            self.dst = img_rot.copy()

    def update(self, cpt=None, width=None, height=None, scale=None, w_scale=None, h_scale=None, rotate=None):
        if width:
            self.width = width
        if height:
            self.height = height
        if scale:
            self.scale = scale
        if w_scale:
            self.w_scale = w_scale
        if h_scale:
            self.h_scale = h_scale
        if cpt:
            self.cpt = cpt
        if rotate:
            self.rotate = rotate
        self.loadImgFile()
        self.setupIMG()
    
    def draw(self, img):
        for callback in self.callback:
            callback()
        # 貼り付け先座標の設定 - alpha_frame がはみ出す場合への対処つき
        position = (round(self.cpt[0] - self.dst.shape[1]/2), round(self.cpt[1] - self.dst.shape[0]/2))
        x1, y1 = max(position[0], 0), max(position[1], 0)
        x2 = min(position[0] + self.dst.shape[1], img.shape[1])
        y2 = min(position[1] + self.dst.shape[0], img.shape[0])
        ax1, ay1 = x1 - position[0], y1 - position[1]
        ax2, ay2 = ax1 + x2 - x1, ay1 + y2 - y1

        # 合成!
        img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * (1 - self.dst[ay1:ay2, ax1:ax2, 3:] / 255) + \
                            self.dst[ay1:ay2, ax1:ax2, :3] * (self.dst[ay1:ay2, ax1:ax2, 3:] / 255)
        return img

class Layer:
    def __init__(self, img=None, width=400, height=300, color=(255, 255, 255), windowname='window'):
        self.objectList = []
        self.setMouseImg('./img/mouse_white.png', 'rgba')
        if img:
            self.base_img = img.copy()
        else:
            self.base_img = makeCanvas(width, height, color=color)
        self.windowname = windowname
        self.callbacklist = []
        self.callback_useSelf_list = []
        self.queue_dict = {}
        self.async_result = None
    
    def __setitem__(self, key, value):
        value.setKey(str(key))
        self.objectList.append(value)

    def __getitem__(self, key):
        for v in self.objectList:
            if v.getKey() == key:
                return v

    def update_baseimg(self, img=None, refresh=True):
        if img:
            self.base_img = img.copy()
        else:
            self.base_img = self.draw()
            if refresh:
                self.objectList = []
    
    def append(self, obj):
        self.objectList.append(obj)

    def setMouseImg(self, path, mode):
        self.mouse = Figure(path, (0, 0), mode=mode)
        self.mouse.setDisplay(False)
    
    def displayMouse(self, flag):
        self.mouse.setDisplay(flag)

    def setQueue(self, key):
        self.queue_dict[key] = queue.Queue()

    def putQueue(self, key):
        if key in self.queue_dict:
            self.queue_dict[key].put()
        else:
            return False, "no key"

    def getQueue(self, key):
        if key in self.queue_dict:
            if not self.queue_dict[key].empty():
                return True, self.queue_dict[key].get()
            else:
                return False, "empty"
        else:
            return False, "no key"
    
    def addCallback(self, func, arg=()):
        self.callbacklist.append({
            "func": func,
            "arg": arg
        })
    
    def addCallback_useSelf(self, func, arg=()):
        self.callback_useSelf_list.append({
            "func": func,
            "arg": arg
        })
    
    def mouseEvents(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse.update(cpt=(x, y))
        for object in self.objectList:
            if object.getEventLisnner() and object.isPtsInner((x, y)):
                for mouseEvent in object.mouseEventCallback:
                    mouseEvent(object, event)

    def draw(self, img=None):
        if img:
            draw_img = img.copy()
        else:
            draw_img = self.base_img.copy()
        for obj in self.objectList:
            if obj.getDisplay():
                obj.draw(draw_img)
        if self.mouse.getDisplay():
            self.mouse.draw(draw_img)
        return draw_img
    
    def isWindowExist(self, windowname):
        try:
            cv2.getWindowProperty(windowname, cv2.WND_PROP_AUTOSIZE)
            return True
        except:
            return False
    
    def run(self, img=None, windowSizeVariable=False, FPS=None, interval=None):
        windowFlag = cv2.WINDOW_NORMAL if windowSizeVariable else cv2.WINDOW_AUTOSIZE
        setinterval = round(1000/FPS) if FPS else 1000
        if interval:
            setinterval = min(setinterval, interval)
        cv2.namedWindow(self.windowname, windowFlag)
        cv2.setMouseCallback(self.windowname, self.mouseEvents)
        while True:
            show_img = self.draw(img)
            cv2.imshow(self.windowname, show_img)
            cv2.waitKey(setinterval)
            # コールバック関数を動作させる
            for func in self.callbacklist:
                func["func"](*func["arg"])
            for func in self.callback_useSelf_list:
                func["func"](self, *func["arg"])
            if not self.isWindowExist(self.windowname):
                break
        cv2.destroyAllWindows()
        return 'finish run'
    
    def run_Async(self, img=None, windowSizeVariable=False, FPS=None, interval=None):
        executor = futures.ThreadPoolExecutor()
        self.async_result = executor.submit(self.run, img, windowSizeVariable, FPS, interval)

    def getResult(self):
        if self.async_result:
            if self.async_result.running():
                return False, "running"
            else:
                return True, self.async_result.result()
        else:
            return False, "no run"