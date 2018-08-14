import sys
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from skimage import exposure,measure,io,img_as_float
from loadData import loadData
import cv2
from skimage import color, restoration
from skimage.color import rgb2gray,gray2rgb
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.restoration import denoise_wavelet
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from PIL import ImageQt, Image,ImageDraw
from dA import dA
from DVAE import DVAE
import pybm3d
import scipy
from scipy import misc,ndimage

from denoise import Ui_MainWindow

class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.picturebutton.clicked.connect(self.set_picture)
        
        self.cutoffslider.valueChanged.connect(self.sigmoid_enh)
        self.gainslider.valueChanged.connect(self.sigmoid_enh)
        self.sigmoid.toggled.connect(self.sigmoid_enh)
        
        self.amountslider.sliderReleased.connect(self.sharpen)
        self.sharpenbox.toggled.connect(self.sharpen)
        
        self.dAbutton.clicked.connect(self.denoise_autoencoder)
        self.bm3dbutton.clicked.connect(self.denoise_bm3d)
        self.originalbutton.clicked.connect(self.show_picture)
        self.dvaebutton.clicked.connect(self.denoise_DVAE)
        
    def browse(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '.')
        fname = open(filename)
        data = fname.read()
        self.textEdit.setText(data)
        fname.close()
        
    def set_picture(self):
        
        filename            = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '.')
        imgpath             = filename[0]
        img                 = Image.open(imgpath)
        
        self.imagelabel.setPixmap(QtGui.QPixmap('original.png'))
        
        img.save('original.png')
        img.save('current.png')
        img.save('current_enh.png')
        
    def sharpen(self):
        
        if self.sharpenbox.isChecked():
            
            w                   = self.amountslider.value()
            
            if self.sigmoid.isChecked():
                img             = np.load('sigmoid.npy')
            else:
                img             = np.load('current.npy')
            
            psf                 = np.ones((w, w)) / 50
            
            deconvolved_RL      = restoration.richardson_lucy(img, psf, iterations=30)
            
            deconvolved_RL      = Image.fromarray(np.uint8(deconvolved_RL*255))
            
            self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(deconvolved_RL)))
    
            deconvolved_RL.save('sharpen.png')
        
        else:
            if self.sigmoid.isChecked():
                self.imagelabel.setPixmap(QtGui.QPixmap('sigmoid.png'))
            else:
                self.imagelabel.setPixmap(QtGui.QPixmap('current.png'))
            
                                     
    def show_picture(self):
               
        img                 = Image.open('original.png')       
               
        self.imagelabel.setPixmap(QtGui.QPixmap('original.png'))

        img.save('current.png')
                        
    def sigmoid_enh(self):
        
        if self.sigmoid.isChecked():
            c                   = self.cutoffslider.value()/100
            g                   = self.gainslider.value()
            
            if self.sharpenbox.isChecked():
                img             = io.imread('sharpen.png', as_grey=True)
            else:
                img             = io.imread('current.png', as_grey=True)
            
            img                 = rgb2gray(img_as_float(img))
            img                 = np.array(img)

            enhanced_img        = exposure.adjust_sigmoid(img, cutoff=c, gain=g, inv=False)    
            np.save('sigmoid',enhanced_img)        
            enhanced_img        = Image.fromarray(np.uint8(enhanced_img*255))
            
            enhanced_img.save('sigmoid.png')
            
            self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(enhanced_img)))
        
        else:
            if self.sharpenbox.isChecked():
                self.imagelabel.setPixmap(QtGui.QPixmap('sharpen.png'))
            else:
                self.imagelabel.setPixmap(QtGui.QPixmap('current.png'))
               
    def denoise_autoencoder(self):
        
        PATCH_SIZE              = 12
        HEIGHT                  = 510
        WIDTH                   = 510
        HUNITS                  = 500
        
        img                     = io.imread('original.png', as_grey=True)
        img                     = rgb2gray(img_as_float(img))
        img                     = np.array(img)
        pat                     = extract_patches_2d(img[0:HEIGHT,0:WIDTH], (PATCH_SIZE,PATCH_SIZE))
        img_patches             = np.reshape(pat,(-1,PATCH_SIZE,PATCH_SIZE))
        
        _,denoised_dA           = dA(img_patches,PATCH_SIZE*PATCH_SIZE,HUNITS,HEIGHT,WIDTH)

        denoised_dA             = np.reshape(denoised_dA.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
        denoised_dA             = reconstruct_from_patches_2d(denoised_dA, (HEIGHT,WIDTH))
       
        np.save('current',denoised_dA)

        enhanced_img            = Image.fromarray(np.uint8(denoised_dA*255))
        
        enhanced_img.save('current.png')
        
        self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(enhanced_img)))
        
    def denoise_DVAE(self):
        
        PATCH_SIZE              = 12
        HEIGHT                  = 510
        WIDTH                   = 510
        HUNITS                  = 200
        HUNITS_mu               = 50
        HUNITS_var              = 50
        
        img                     = io.imread('original.png', as_grey=True)
        img                     = rgb2gray(img_as_float(img))
        img                     = np.array(img)
        pat                     = extract_patches_2d(img[0:HEIGHT,0:WIDTH], (PATCH_SIZE,PATCH_SIZE))
        img_patches             = np.reshape(pat,(-1,PATCH_SIZE,PATCH_SIZE))
        
        denoised_DVAE           = DVAE(img_patches,PATCH_SIZE*PATCH_SIZE,HUNITS,HUNITS_mu,HUNITS_var)

        denoised_DVAE            = np.reshape(denoised_DVAE.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
        denoised_DVAE             = reconstruct_from_patches_2d(denoised_DVAE, (HEIGHT,WIDTH))
       
        np.save('current',denoised_DVAE)

        enhanced_img            = Image.fromarray(np.uint8(denoised_DVAE*255))
        
        enhanced_img.save('current.png')
        
        self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(enhanced_img)))
    
    def denoise_bm3d(self):
        
        img                     = io.imread('original.png', as_grey=True)
        img                     = rgb2gray(img_as_float(img))
        img                     = np.array(img)
        
        denoised_bm3d           = pybm3d.bm3d.bm3d(img, 0.1)
        
        np.save('current',denoised_bm3d)
        
        enhanced_img            = Image.fromarray(np.uint8(denoised_bm3d*255))

        enhanced_img.save('current.png')
        
        self.imagelabel.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(enhanced_img)))
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
    
    
    
    
    