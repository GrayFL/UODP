import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from torchvision.ops import nms
import torch

def GetCenter(pts):
    pts = np.array(pts)
    x = np.mean(pts[:,0])
    y = np.mean(pts[:,1])
    return [x,y]

def mouse_click(event, x, y, flags, param):
    img,G,times = param[0],param[1],param[2]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x={x},y={y},G={G[y,x]}')
        #print(f'[{x},{y}],',end='')
        times[0]+=1

def Gray_diff(m1,m2):
    m1 = m1.astype('int16')
    m2 = m2.astype('int16')
    mr = m1-m2
    res = np.abs(mr)
    return res.astype('uint8')
    
def Lab_diff(m1,m2):
    m1 = m1.astype('float32')
    m2 = m2.astype('float32')
    mr = m1-m2
    mr1,mr2,mr3 = cv2.split(mr)
    res = np.sqrt(mr1**2+mr2**2+mr3**2)*255/441.7
    return res.astype('uint8')

def getGred(img,c = 2,fun = Gray_diff):
    img_h,img_w = img.shape[0]-c,img.shape[1]-c
    vgap = fun(img[c:,c:],img[:-c,c:])
    hgap = fun(img[c:,c:],img[c:,:-c])
    hogn = ((vgap+hgap)//2).astype('uint8')
    # hogn = cv2.bitwise_not(hogn)
    return hogn

def getGred_Filter(img,kernel):
    hogn = cv2.filter2D(img,-1,kernel = kernel)
    return hogn

def BGR2S_HQ_weight(img):
    img_max = np.max(img,axis = 2).astype('float')
    img_min = np.min(img,axis = 2).astype('float')
    S = (img_max-img_min)/(img_max+0.00001)
    return S

def BGR2S_HQ(img):
    img_max = np.max(img,axis = 2).astype('float')
    img_min = np.min(img,axis = 2).astype('float')
    S = (img_max-img_min)/img_max*255
    return S.astype('uint8')

def BGR2V_HQ(img):
    img_max = np.max(img,axis = 2).astype('float')
    V = img_max
    return V.astype('uint8')

def BGR2SV_HQ(img):
    img_max = np.max(img,axis = 2).astype('float')
    img_min = np.min(img,axis = 2).astype('float')
    S = (img_max-img_min)/img_max*255
    V = img_max
    SV = (S+V)/2
    return SV.astype('uint8')
            
def set_CV2_axes(ax):
    ax.xaxis.set_ticks_position('top') # 将x轴的位置设置在顶部
    ax.invert_yaxis() # y轴反向
    
# 高斯核生成函数
def creat_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    gauss /= np.sum(gauss)
    return gauss

def diff_pic(img1,img2):
    res = img1.astype('int16')-img2.astype('int16')
    res = np.abs(res).astype('uint8')
    return res
def minus_pic(img1,img2):
    res = img1.astype('int16')-img2.astype('int16')
    res[res<0] = 0
    res = res.astype('uint8')
    return res

def getFeature_PCA(img):
    h,w = img.shape[0],img.shape[1]
    new_pix = img.reshape((-1,3))
    pca = PCA(n_components=1)
    pca.fit(new_pix)
    new_data = pca.transform(new_pix)
    new_data = new_data*0.5+128
    new_img = new_data.reshape((h,w)).astype('uint8')
    return new_img

def mark_on_pic(src_img,pts,color):
    img = copy.deepcopy(src_img)
    size = 3
    for pt in pts:
        cv2.circle(img,(pt[1],pt[0]),size,color,-1)
    # img[pts] = color
    return img

class spirit_pic():
    def __init__(self,src,test_pt,r):
        self.mark = test_pt
        self.r = r
        self.angle = None
        self.index = None
        self.offset = None
        self.bound = None
        self.boundN = None
        self.zeros = None
        self.score = None
        self.img = self.getSurround(src,r)
    
    def __str__(self):
        return f'{self.img}'

    def getSurround(self,src,r):
        pt = self.mark
        img = src
        img_h,img_w = img.shape[0],img.shape[1]
        h0,h1 = np.max([0,pt[0]-r]),np.min([img_h,pt[0]+r])
        w0,w1 = np.max([0,pt[1]-r]),np.min([img_w,pt[1]+r])
        h = h1-h0
        w = w1-w0
        self.zeros = np.array([h0,w0])
        # shape = [h,w]
        # i = np.argmin(shape)
        # index = pt[i]-self.zeros[i]
        index = pt-self.zeros
        self.index = index
        img = copy.deepcopy(img[h0:h1+1,w0:w1+1])
        self.img = img
        return self.img

    def getAngle(self,img_shape,center = None):
        test_pt = self.mark
        angle = 0
        if center is None:
            angle = (test_pt[1]-img_shape[1]/2)/(test_pt[0]-img_shape[0]/2)
        else:
            if test_pt[0]-center[0] == 0:
                angle = img_shape[1]
            else:
                angle = (test_pt[1]-center[1])/(test_pt[0]-center[0])
        # if np.isinf(angle) or np.isnan(angle):
        #     angle = img_shape[1]
        self.angle = angle
        self.offset = self.r*(1-angle)
        return self.angle
    
    def Filter(self,kernel,replace=False):
        temp = self.img
        res = cv2.filter2D(temp,-1,kernel=kernel)
        if replace:
            self.img = res
            return self
        else:
            new = copy.deepcopy(self)
            new.img = res
            return new
    def Gauss_Filter(self,ksize,replace=False):
        temp = self.img
        res = cv2.GaussianBlur(temp, (ksize,ksize), 1)
        if replace:
            self.img = res
            return self
        else:
            new = copy.deepcopy(self)
            new.img = res
            return new
    
    def Find_Edge(self,minVal=100,maxVal=255,replace=False):
        edge = cv2.Canny(self.img,minVal,maxVal)
        if replace:
            self.img = edge
            return self
        else:
            new = copy.deepcopy(self)
            new.img = edge
            return new
        
    def Search_Bound(self,ksize=3,threshold_bbox=0.8,minVal=100,maxVal=255):
        img = self.img
        h,w = img.shape[0],img.shape[1]
        pt = self.mark
        angle = self.angle
        index_offset = -1
        # print(f'angle={self.angle},offset={self.offset}')
        
        if abs(angle)<=0.5:
            index_start = self.r
            sh = np.arange(self.r*2,dtype='int')
            sw = sh*self.angle+self.offset
            sp = []
            # print(sh,sw,h,w)
            for i in range(self.r*2):
                if h>sh[i]>=0 and w>sw[i]>=0:
                    if index_offset == -1:
                        index_offset = i
                    sp.append([sh[i],sw[i]])
            sp = np.array(sp,dtype='int')
            sh,sw = sp[:,0],sp[:,1]
            
        elif abs(angle)>0.5:
            index_start = self.r
            sw = np.arange(self.r*2,dtype='int')
            sh = (sw-self.offset)/self.angle
            sp = []
            # print(sh,sw,h,w)
            for i in range(self.r*2):
                if h>sh[i]>=0 and w>sw[i]>=0:
                    if index_offset == -1:
                        index_offset = i
                    sp.append([sh[i],sw[i]])
            sp = np.array(sp,dtype='int')
            sh,sw = sp[:,0],sp[:,1]
            
        # search_line = img[sh,sw]
        # search_line_m = cv2.Canny(search_line,minVal,maxVal)
        img = cv2.Canny(img,minVal,maxVal)
        search_line_m = img[sh,sw]
        search_line_m = cv2.blur(search_line_m,(ksize,ksize))
        search_line_m = search_line_m.reshape(-1)
        
        # plt.close()
        # fig = plt.figure(figsize=(10,6),dpi=100)
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(search_line)
        # ax.plot(search_line_m)
        # plt.show()
        
        # loc = np.where(search_line_m>0)
        length = len(search_line_m)
        bound = [0,length-1]
        i = index_start-index_offset
        # print(self.index)
        # print('start',index_start,'offset',index_offset)
        for j in range(0,i+1):
            if search_line_m[i-j]>0:
                for k in range(0,i-j+1):
                    if search_line_m[i-j-k]==0:
                        bound[0] = i-j-k
                        break
                break
        for j in range(0,length-i):
            if search_line_m[i+j]>0:
                # print(f'j={j},i={i},search_line_m[i+j]={search_line_m[i+j]}')
                for k in range(0,length-i-j):
                    # print(f'k={k},id={i+j+k},search_line_m[i+j]={search_line_m[i+j+k]}')
                    if search_line_m[i+j+k]==0:
                        # print('in')
                        bound[1] = i+j+k
                        break
                break
        # print(bound)
        self.img[sh,sw] = [255]
        if abs(bound[1]-bound[0])>(np.max(self.img.shape)*threshold_bbox):
            return None
        x1,y1,x2,y2 = np.min(sw[bound]),np.min(sh[bound]),\
                      np.max(sw[bound]),np.max(sh[bound])
        
        min_ratio = 0.7
        if y2-y1 < self.r*min_ratio:
            fix = int((self.r*min_ratio-(y2-y1))/2)
            y1,y2 = np.max([0,y1-fix]),np.min([self.img.shape[0],y2+fix])
        if x2-x1 < self.r*min_ratio:
            fix = int((self.r*min_ratio-(x2-x1))/2)
            x1,x2 = np.max([0,x1-fix]),np.min([self.img.shape[1],x2+fix])
        if abs(angle)<=0.5 and (x2-x1)<(y2-y1)*0.5:
            fix = ((y2-y1)//2-(x2-x1))//2
            x1,x2 = np.max([0,x1-fix]),np.min([self.img.shape[1],x2+fix])
        if abs(angle)>0.5 and (x2-x1)*0.5>(y2-y1):
            fix = ((x2-x1)//2-(y2-y1))//2
            y1,y2 = np.max([0,y1-fix]),np.min([self.img.shape[0],y2+fix])
        
        self.bound = np.array([[x1,y1],
                               [x2,y2]])
        self.boundN = np.array([[x1+self.zeros[1],y1+self.zeros[0]],
                               [x2+self.zeros[1],y2+self.zeros[0]]])
        self.score = -4*(np.max([y2-y1,x2-x1])/np.max([h,w])-0.5)**2+1
        return self.boundN

def detect(img,
           ksize_bed=65,
           ksize_sp=1,
           ksize_find=4,
           img_center=(0,0),
           restrain=0,
           threshold_det=100,
           threshold_iou=0.1,
           threshold_bbox=0.9,
           canny_min=100,
           canny_max=255,
           radius_bound=32,
           db_eps=10,
           db_samples=10,
           db_sample_ratio=100,
           linewidth=1,
           debug=False
          ):
    """
    ksize_bed: K_bg
    ksize_sp: K_dn
    ksize_find: K_bl
    threshold_det: T_FFL
    threshold_iou: T_iou
    threshold_bbox: T_bma
    radius_bound: R
    db_eps: C_ep
    db_samples: C_ms
    debug: When "Debug" is True, the images of FFL, FFL', Marker, etc. during processing will be output to assist in parameterization.
    ---

    Here are some preset parameter settings:

    Params_set_A(DEFAULT):
        ksize_bed(K_bg) = 65
        ksize_sp(K_dn) = 1
        ksize_find(K_bl) = 4
        threshold_det(T_FFL) = 100
        threshold_iou(T_iou) = 0.1
        threshold_bbox(T_bma) = 0.9
        radius_bound(R) = 32
        db_eps(C_ep) = 10
        db_samples(C_ms) = 10
    
    Params_set_B:
        ksize_bed(K_bg) = 20
        ksize_sp(K_dn) = 3
        ksize_find(K_bl) = 7
        threshold_det(T_FFL) = 80
        threshold_iou(T_iou) = 0.05
        threshold_bbox(T_bma) = 0.8
        radius_bound(R) = 25
        db_eps(C_ep) = 6
        db_samples(C_ms) = 5

    Params_set_C:
        ksize_bed(K_bg) = 42
        ksize_sp(K_dn) = 1
        ksize_find(K_bl) = 7
        threshold_det(T_FFL) = 55
        threshold_iou(T_iou) = 0.1
        threshold_bbox(T_bma) = 0.4
        radius_bound(R) = 32
        db_eps(C_ep) = 14
        db_samples(C_ms) = 50
        db_sample_ratio = 3
    """
    img = cv2.resize(img,(0,0),fx=1,fy=1)
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_shape = (img_h,img_w)
    
    # 提取lab空间的亮度特征fet
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    fet,_,_ = cv2.split(lab)
    fet_s = BGR2S_HQ_weight(img)
    fet_bank = fet
    fet = (fet*(fet_s)**restrain).astype('uint8')
    # det = np.ones((img_shape),dtype='uint8')*255

    # 提取img的背景bed
    # bed = cv2.GaussianBlur(fet, (ksize_bed,ksize_bed), 1)
    kernelG = creat_gauss_kernel(ksize_bed)
    bed = getGred_Filter(fet,kernelG)
    
    # 获得比背景更黑的区域，并取点坐标
    det = minus_pic(bed,fet)
    det2 = det > threshold_det
    data_det = np.argwhere(det > threshold_det)
    data_det = np.hstack([data_det,det[det > threshold_det].reshape((-1,1))])
    
    # 三维聚类
    model_dbs = DBSCAN(eps = db_eps, min_samples = db_samples)
    model_dbs.fit(data_det)
    labels = model_dbs.labels_
    n_clusters = len(set(labels))-1
    centers = []
    
    # plt.close()
    # fig = plt.figure(figsize=(12,6),dpi=100)
    # ax = fig.add_subplot(1,1,1)
    # ax.set_xlim((0,img_w))
    # ax.set_ylim((0,img_h))
    # set_CV2_axes(ax)
    
    for i in range(0,n_clusters):
        one_cluster = data_det[labels == i]
        if len(one_cluster)<=db_samples*db_sample_ratio:
            centers.append(GetCenter(one_cluster))
        # ax.scatter(one_cluster[:,1],one_cluster[:,0],s=2)

    # plt.show()

    centers = np.array(centers,dtype='int')
    bboxes = np.zeros((len(centers),4),dtype='int')
    scores = np.zeros(len(centers))
    for i,test_pt in enumerate(centers):
        # print(test_pt)
        test_pic = spirit_pic(img,test_pt,radius_bound)
        test_pic.getAngle(img_shape,img_center)

        ksize_sp = ksize_sp//2*2+1
        test_pic.Gauss_Filter(ksize_sp,replace=True)
        test_pic.img = cv2.cvtColor(test_pic.img,cv2.COLOR_BGR2GRAY)

        if test_pic.Search_Bound(ksize=ksize_find,\
                                 threshold_bbox=threshold_bbox,\
                                 minVal=canny_min,\
                                 maxVal=canny_max) is None: continue

        # res = cv2.rectangle(res,lap_pic.boundN[0],lap_pic.boundN[1],color=(0,255,255))
        bboxes[i] = test_pic.boundN.reshape(-1)
        scores[i] = test_pic.score

    # res = copy.deepcopy(img)
    # res = mark_on_pic(img,centers,(0,255,255))
    keep = nms(torch.tensor(bboxes,dtype = torch.float),
               torch.tensor(scores,dtype = torch.float),
               iou_threshold = threshold_iou)
    out = []
    for i in keep:
        # res = cv2.rectangle(res,
        #                     bboxes[i][0:2],
        #                     bboxes[i][2:4],
        #                     color=(0,255,255),
        #                     thickness=linewidth)
        out.append(bboxes[i][0:4])
    mark = mark_on_pic(img,centers,(0,255,255))
    out = np.array(out)
    if debug:
        return out,[det,det2,data_det,mark,fet_bank,fet,(fet_s)]
    else:
        return out