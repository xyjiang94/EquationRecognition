from skimage import util, measure, io

class Segmentation:
    def __init__(self,image_path):
        img = io.imread(image_path)
        self.image = util.img_as_ubyte(img)

    def segment(self):
        labeled_image = measure.label(self.image)
        regionProperties = measure.regionprops(labeled_image,self.image)
        for e in regionProperties:
            if e.area>20:
                print "area",e.area
                print "bbox",e.bbox
                print "-----------------------"
                io.imsave("segments/"+str(e.bbox)+".png",self.image[e.bbox[0]:e.bbox[2],e.bbox[1]:e.bbox[3]])

seg = Segmentation('equations/SKMBT_36317040717260_eq7.png')
seg.segment()
