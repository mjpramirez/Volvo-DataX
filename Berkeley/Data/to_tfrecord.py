#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import hashlib
import pickle

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('image_list', './data/voc2012_raw/VOCdevkit/VOC2012/', 'path to img_list')
flags.DEFINE_string('annotations', './data/voc2012_raw/VOCdevkit/VOC2012/', 'path to annotations')
#flags.DEFINE_enum('split', 'train', ['train', 'val'], 'specify train or val split')
flags.DEFINE_string('output_file', './data/voc2012_train.tfrecord', 'output dataset')
#flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')


def build_example(annotation):
    
    img_path = annotation[0]
    img_raw = open(img_path, 'rb').read()
    #key = hashlib.sha256(img_raw).hexdigest()

    width = 1920
    height = 1080

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    #classes = []
    classes_text = []
    #truncated = []
    #views = []
    #difficult_obj = []
    
    
    if len(annotation)> 2:
        for i in range(1,(len(annotation)-1)):
            #difficult = bool(int(obj['difficult']))
            #difficult_obj.append(int(difficult))

            xmin.append(float(int(annotation[i].split(',')[0])) / width)
            ymin.append(float(int(annotation[i].split(',')[1])) / height)
            xmax.append(float(int(annotation[i].split(',')[2])) / width)
            ymax.append(float(int(annotation[i].split(',')[3])) / height)
            classes_text.append('pedestrian'.encode('utf8'))
            #classes.append(class_map[obj['name']])
            #truncated.append(int(obj['truncated']))
            #views.append(obj['pose'].encode('utf8'))

            # only the following features are required for yolo encoding
            
    example = tf.train.Example(features=tf.train.Features(feature={
        #'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        #'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        #'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #    annotation['filename'].encode('utf8')])),
        #'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #    annotation['filename'].encode('utf8')])),
        #'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        #'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        #'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        #'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
       # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        #'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


'''def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}'''


def main(_argv):
    #class_map = {name: idx for idx, name in enumerate(open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded")

    writer = tf.io.TFRecordWriter(FLAGS.output_file)

    
#    image_list = open(os.path.join(
#        FLAGS.data_dir, 'ImageSets', 'Main', 'aeroplane_%s.txt' % FLAGS.split)).read().splitlines()

    with open(FLAGS.image_list, "rb") as fp:   # Unpickling
        image_list = pickle.load(fp)   
        
    with open(FLAGS.annotations, "rb") as fp1:   # Unpickling
        annotations = pickle.load(fp1)    
        
    logging.info("Image list loaded: %d", len(image_list))
    
    for idx,image in enumerate(tqdm.tqdm(image_list)):
#        name, _ = image.split()
#        annotation_xml = os.path.join(
#            FLAGS.data_dir, 'Annotations', name + '.xml')
#        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
#        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotations[idx])
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)

