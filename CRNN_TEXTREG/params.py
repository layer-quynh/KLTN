from CRNN_TEXTREG.tool import alphabets

alphabet = alphabets.alphabet
keep_ratio = False
manualSedd = 1234
random_sample = True
height = 32
width = 128
number_hidden = 256
n_channel = 1

pretrained = ''
expr_dir = 'expr'
dealwith_lossnan = False

cuda = True
multi_gpu = False
ngpu = 1
workers = 0

# train process
displayInterval = 100
valInterval = 1000
saveInterval = 1000
n_val_disp = 10

#finetune
nepoch = 1000
batchSize = 2
lr = 0.0001
beta1 = 0.5
adam = False
adadelta = False