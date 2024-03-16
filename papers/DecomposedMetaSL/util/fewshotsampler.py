
'''
The code is adapted from https://github.com/thunlp/Few-NERD/blob/main/util/fewshotsampler.py
'''
import random
import numpy as np
import json

class FewshotSampleBase:
    '''
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    '''
    def __init__(self):
        self.class_count = {}

    def get_class_count(self):
        '''
        return a dictionary of {class_name:count} in format {any : int}
        '''
        return self.class_count


class FewshotSampler:
    '''
    sample one support set and one query set
    '''
    def __init__(self, N, K, Q, samples, classes=None, random_state=0):
        '''
        N: int, how many types in each set
        K: int, how many instances for each type in support set
        Q: int, how many instances for each type in query set
        samples: List[Sample], Sample class must have `get_class_count` attribute
        classes[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        random_state[Optional]: int, the random seed
        '''
        self.K = K
        self.N = N
        self.Q = Q
        self.samples = samples
        self.__check__() # check if samples have correct types
        if classes:
            self.classes = classes
        else:
            self.classes = self.__get_all_classes__()
        random.seed(random_state)

    def __get_all_classes__(self):
        classes = []
        for sample in self.samples:
            classes += list(sample.get_class_count().keys())
        return list(set(classes))

    def __check__(self):
        for idx, sample in enumerate(self.samples):
            if not hasattr(sample,'get_class_count'):
                print('[ERROR] samples in self.samples expected to have `get_class_count` attribute, but self.samples[{idx}] does not')
                raise ValueError

    def __additem__(self, index, set_class):
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(self, sample, set_class, target_classes):
        threshold = 2 * set_class['k']
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                return False
            if class_count[class_name] + set_class.get(class_name, 0) > threshold:
                return False
            if set_class.get(class_name, 0) < set_class['k']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class):
        if len(set_class) < self.N+1:
            return False
        for k in set_class:
            if set_class[k] < set_class['k']:
                return False
        return True

    def __get_candidates__(self, target_classes):
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]


    def __next__(self):
        '''
        randomly sample one support set and one query set
        return:
        target_classes: List[any]
        support_idx: List[int], sample index in support set in samples list
        support_idx: List[int], sample index in query set in samples list
        '''
        support_class = {'k':self.K}
        support_idx = []
        query_class = {'k':self.Q}
        query_idx = []
        target_classes = random.sample(self.classes, self.N)
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        # greedy search for support set
        while not self.__finish__(support_class):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.__valid_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
        # same for query set
        while not self.__finish__(query_class):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.__valid_sample__(self.samples[index], query_class, target_classes):
                    self.__additem__(index, query_class)
                    query_idx.append(index)
        return target_classes, support_idx, query_idx

    def __iter__(self):
        return self

class DebugSampler:
    def __init__(self, filepath, querypath=None):
        data = []
        with open(filepath, mode="r", encoding="utf-8") as fp:
            for line in fp:
                data.append(json.loads(line))
        self.data = data
        if querypath is not None:
            with open(querypath, mode="r", encoding="utf-8") as fp:
                self.ment_data = [json.loads(line) for line in fp]
            if len(self.ment_data) != len(self.data):
                print("the mention data len is different with input episode number!!!!!!!!!!!!!!!")
        else:
            self.ment_data = None
        return
        

    def __next__(self, idx):
        idx = idx % len(self.data)
        target_classes = self.data[idx]["target_classes"]
        support_idx = self.data[idx]["support_idx"]
        query_idx = self.data[idx]["query_idx"]
        if self.ment_data is None:
            return target_classes, support_idx, query_idx
        return target_classes, support_idx, query_idx, self.ment_data[idx]

    def __len__(self):
        return len(self.data)
    