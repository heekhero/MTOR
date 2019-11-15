
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

import codecs
import os
import cv2

XML_EXT = '.xml'

class LabelFile(object):
    # It might be changed as window creates
    suffix = '.lif'

    def __init__(self, filename, imagePath, classes):
        assert(os.path.exists(imagePath))
        self.shapes = ()
        self.classes = classes
        self.imagePath = imagePath
        self.filename = filename
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        self.imageShape = image.shape

    def savePascalVocFormat(self, dets):
        imgFolderPath = os.path.dirname(self.imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(self.imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                                 self.imageShape, localImgPath=self.imagePath)

        for cls in self.classes:
            if cls in dets:
                for bbox in dets[cls]:
                    bbox = self.prettifyBndBox(bbox)
                    writer.addBndBox(bbox, cls)

        writer.save(targetFile=self.filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified

    def prettifyBndBox(self, bbox):
        xmin, ymin, xmax, ymax = bbox[:4]
        height, width, _ = self.imageShape
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        if xmax > width:
            xmax = width

        if ymax > height:
            ymax = height

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix


class PascalVocWriter:
    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)
        # return etree.tostring(root, pretty_print=False)

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        # top.set('verified', 'yes' if self.verified else 'no')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        localImgPath = SubElement(top, 'path')
        localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, bbox, name):
        bndbox = {'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3]}
        bndbox['name'] = name
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = "0"
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
            # prob = SubElement(object_item, 'prob')
            # prob.text = '{:.3f}'.format(each_object['prob'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()