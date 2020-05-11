from CRNN_TEXTREG.model.model import CRNN
import CRNN_TEXTREG.params as params
import torch
from CRNN_TEXTREG.tool import dataset
from CRNN_TEXTREG import utils
from torch.autograd import Variable
from PIL import Image
import os
import time
import numpy as np
from sklearn.preprocessing import normalize

class TEXTREG(object):
    def __init__(self, path_weights="/home/quynhnguyen/Documents/project/uet/khoa_luan/make_mutext_img/weights/text_reg/weights-1.pth"):
        self.model = CRNN(params.height, params.n_channel, len(params.alphabet) + 1, params.number_hidden)
        self.model.to(torch.device("cpu"))
        self.model.load_state_dict(torch.load(path_weights, map_location="cpu"))
        self.converter = utils.strLabelConverter(params.alphabet)
        self.transformer = dataset.resizeNormalize((128, 32))
        self.model.eval()
        self.dict_text = {}
        for index, char in enumerate(params.alphabet):
            self.dict_text[char] = index
        
    def predict(self, img):
        image = self.transformer(img)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred

    def predict_batch(self, list_img):
        list_batch = [torch.Tensor(self.transformer(img)) for img in list_img]
        batch_pred = torch.cat(list_batch)
        batch_pred = batch_pred.view(len(list_img), 1, batch_pred.size(1), batch_pred.size(2))
        image = Variable(batch_pred)
        preds = self.model(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * len(list_img)))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        return " ".join(sim_preds)

# model = TEXTREG()
# path_test_data = "/home/quynhnguyen/Documents/project/uet/khoa_luan/data/4"
# files = [os.path.join(path_test_data, f) for f in os.listdir(path_test_data) if os.path.isfile(os.path.join(path_test_data, f))]
# # print(files)
# i = 1
# predict_correct = 1
# for file in files:
#     # real_word = file.split("_", 2)[1]
#     _, real_word = file
#     print(real_word)
#     pred_word = model.predict(Image.open(file).convert("L"))
#     if real_word == pred_word:
#         predict_correct += 1
#
# accuracy = (predict_correct / float(len(files))) * 100
# print("Accuracy: ", accuracy)
# print("Number images of test data: ", len(files))