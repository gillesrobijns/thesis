from skimage import exposure,measure
import sys 

def sigmoid(image,cutoff,gain):

    enh_img = exposure.adjust_sigmoid(image, cutoff, gain, inv=False)

    return enh_img

if __name__ == '__main__':

    image       = float(sys.argv[1])
    cutoff      = float(sys.argv[2])
    gain        = float(sys.argv[3])

    sys.stdout.write(str(sigmoid(image,cutoff,gain)))

