# define dataset class to feed the model
import numpy as np 
import os
import sys
import time
import importlib
from collections import OrderedDict, Mapping, Container
from pprint import pprint

from sys import getsizeof



def build(mtype, kwargs):
    module_name, cls_name = mtype.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    if kwargs is not None:
        return cls(**kwargs)
    else:
        return cls()

def ModelLoader(cfg):
    cfg_subnet = cfg
    mtype = cfg_subnet['type']
    kwargs = cfg_subnet['kwargs']
    module = build(mtype, kwargs)
    return module

class DepLoader(object):
    def __init__(self,cfg,debug=False):
        self.cfg = cfg
        self.debug = debug
        for key in self.cfg:
            self.__build_module(key)
            
    def __build_module(self,key):
        # 如果模块已经build过，直接返回就好
        if getattr(self,key,None) is not None:
            return getattr(self,key,None)
        
        # 如果没有build过，那么先build这个模块的依赖
        one_cfg = self.cfg[key]
        deps = one_cfg.get('dependence',[])
        dep_dic = {}
        for one_dep in deps:
            dep_dic[one_dep] = self.__build_module(one_dep)
            
        mtype = one_cfg['type']
        
        kwargs = one_cfg.get('kwargs',{})
        if kwargs == None:
            kwargs = {}
        
        kwargs.update(dep_dic)
        if self.debug:
            print("building:",key," params: ",list(kwargs.keys()))
        
        if len(kwargs) == 0:
            kwargs = None
        module = build(mtype, kwargs)
        
        setattr(self,key,module)
        return module

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
        
class Dataset():
    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        assert(data.shape[0] == label.shape[0])
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end],self._label[start:end]

class ProgressBar():
    def __init__(self,worksum,info="",auto_display=True):
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display
    def startjob(self):
        self.begin_time = time.time()
    def complete(self,num):
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display == True:
            self.display_progress_bar()
    def display_progress_bar(self):
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
        str_log = ("%s %.2f %% %s %s/%s \t used:%ds eta:%d s" % (self.info,percent,strprogress,self.finishsum,self.worksum,self.gaptime,eta_time))
        sys.stdout.write('\r' + str_log)

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def deep_compare(a, b, pointer='/'):
    if a == b:
        return

    if type(a) != type(b):
        reason = 'Different data types'
        extra = str((type(a), type(b)))
        x(pointer, reason, extra)

    elif type(a) in (set, frozenset):
        pointer += 'set()'
        if len(a) != len(b):
            pointer += 'set()'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        reason = 'Different items'
        extra = (a, b)
        x(pointer, reason, extra)

        for i in range(len(a)):
            deep_compare(a[i], b[i], pointer + 'set()'.format(i))

    elif type(a) in (list, tuple):
        if len(a) != len(b):
            pointer += '[]'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        if sorted(a) == sorted(b):
            pointer += '[]'
            reason = 'Different sort order'
            extra = 'N/A'
            x(pointer, reason, extra)

        for i in range(len(a)):
            deep_compare(a[i], b[i], pointer + '[{}]'.format(i))

    elif type(a) in (dict, OrderedDict):
        if len(a) != len(b):
            pointer += '{}'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        if set(a.keys()) != set(b.keys()):
            pointer += '{}'
            reason = 'Different keys'
            extra = (a.keys(), b.keys())
            x(pointer, reason, extra)

        for k in a:
            deep_compare(a[k], b[k], pointer + '[{}]'.format(k))
    else:
        reason = 'Different objects'
        extra = (a, b)
        x(pointer, reason, extra)


def x(pointer, reason, extra):
    message = 'Objects are not the same. Pointer: {}. Reason: {}. Extra: {}'
    raise RuntimeError(message.format(pointer, reason, extra))


def compare(a, b):
    try:
        deep_compare(a, b, '/')
    except RuntimeError as e:
        pprint(e.message)


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
    This is a recursive function that rills down a Python object graph
    like a dictionary holding nested ditionaries with lists of lists
    and tuples and sets.
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r