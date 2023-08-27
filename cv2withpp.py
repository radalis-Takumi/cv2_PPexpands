import cv2
import numpy as np
import math

def makeCanvas(width, height, color=(255, 255, 255)):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, -1)
    return img

class Rectangle:
    def __init__(self, cpt, width, height, rad=0, rotate=0, fillcolor=None, framecolor=None):
        self.cpt = cpt
        self.width = width
        self.height = height
        self.rad = rad
        self.rotate = rotate
        self.fillcolor = fillcolor
        self.framecolor = framecolor
        self.para_change(rad=self.rad)
        
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
        radians = math.radians(rotate)
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
            self.obj_radChange(rad)
        if rotate:
            self.obj_rotete(rotate)
        else:
            self.obj_rotete(self.rotate)
        if cpt:
            self.obj_move(cpt)
        else:
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

class Layer:
    def __init__(self):
        self.objectList = []
    
    def append(self, obj):
        self.objectList.append(obj)

    def draw(self, img):
        for obj in self.objectList:
            obj.draw(img)
        return img