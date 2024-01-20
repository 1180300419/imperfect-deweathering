'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-09-27 12:50:49
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-16 16:35:17
'''
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np


def find_dataset_using_name(dataset_name, split='train'):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of "
                        "BaseDataset with class name that matches %s in "
                        "lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def create_dataset(dataset_name, split, opt):
    data_loader = CustomDatasetDataLoader(dataset_name, split, opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
	def __init__(self, dataset_name, split, opt):
		self.opt = opt
		dataset_class = find_dataset_using_name(dataset_name, split)
		self.dataset = dataset_class(opt, split, dataset_name)
		self.imio = self.dataset.imio
		self.batch_size = opt.batch_size if ('train' in split) else 1
		print("dataset [%s(%s)] created" % (dataset_name, split))
		scene, scene_indices = self.dataset.get_scene_indices()
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_sampler=CustomBatchSampler(
				scene,
				scene_indices,
				batch_size=self.batch_size if split == 'train' else 1),
				num_workers=opt.num_dataloader,
				pin_memory=True
			)

	def load_data(self):
		return self.dataloader


import copy
import random


class CustomBatchSampler():
	def __init__(self, scene, scene_indices, batch_size=8):
		self.scene_indices = scene_indices
		self.batch_size = batch_size
		self.scene = scene
		self.num_batches = int((scene_indices[-1][-1] + 1) / batch_size)

	def __len__(self):
		return self.num_batches

	def __iter__(self):
		scene_indices = copy.deepcopy(self.scene_indices)
		scene = copy.deepcopy(self.scene)

		for scene_list in scene_indices:
			random.shuffle(scene_list)  # 随机打乱同一个场景中的雨图

		out_indices = []
		done = False

		while not done:
			out_batch_indices = []
			if (len(scene_indices) < self.batch_size):
				# print(len(scene_indices))
				self.num_batches = len(out_indices)
				return iter(out_indices)

			chosen_scenes = np.random.choice(len(scene_indices), len(scene_indices), replace = False)
			
			chosen_indexes = []
			count = 0
			for index in chosen_scenes:  # 目前选中的场景下标
				flag = True
				for tmp_index in chosen_indexes:  # 已经选中的场景下标
					if scene[index][:-4] == scene[tmp_index][:-4]:
						flag = False
						break
				if flag:
					chosen_indexes.append(index)
					count += 1
				if count == self.batch_size:
					break

			if len(chosen_indexes) < self.batch_size:
				return iter(out_indices)

			chosen_scenes = chosen_indexes  # 目前选中的场景下标
			empty_indices = []
			for i in chosen_scenes:
				scene_list = scene_indices[i]  # 记录当前场景中的图片列表
				out_batch_indices.append(scene_list.pop())
				if (len(scene_list) == 0):
					empty_indices.append(i)
			empty_indices.sort(reverse=True)
			for i in empty_indices:
				scene_indices.pop(i)
				scene.pop(i)
			out_indices.append(out_batch_indices)
		self.num_batches = len(out_indices)
		return iter(out_indices)