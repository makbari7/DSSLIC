### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import time

from torch.autograd import Variable

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
for i, data in enumerate(dataset):    
    if i >= opt.how_many:
        break
             
    start_time = time.time()
    generated , res, comp_image, up_image = model.inference(data['image'],data['label'], data['ds'])
    print("--- %s seconds ---" % (time.time() - start_time))
    visuals = OrderedDict([
      ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),                           
                           ('synthesized_image', util.tensor2im(generated.data[0])),
			   ('comp_image', util.tensor2im(comp_image.data[0])),
                           ('fine_image', util.tensor2im(res.data[0])),
                           ('real_image', util.tensor2im(data['image'][0])),
                           ('up_image', util.tensor2im(up_image.data[0]))
                           ])    
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
