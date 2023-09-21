import torchvision.models as models                           # for example model
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, Academic etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.advanced_ptq import ptq_reconstruction
import torch

from mqbench.data.imagenet import build_imagenet_data
from tqdm import tqdm
from mqbench.utils.logger import logger
from mqbench.data.utils import seed_all
from mqbench.data.evaluate import evaluate


##### presetting ####
def load_calibrate_data(train_loader, cali_batchsize):
    calib_data = []
    for i, batch in tqdm(enumerate(train_loader)):
        calib_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return calib_data

batch_size, num_workers, data_path = 32, 4, 'C:/Users/vjuneja/Downloads/imagenet-mini'
train_loader, val_loader = build_imagenet_data(batch_size=batch_size, workers=num_workers, data_path=data_path)
device = "cpu"
calib_batch_size, ptq_batch_size = 24, 32
print("loading data")
calib_data = load_calibrate_data(train_loader, calib_batch_size)
ptq_data = load_calibrate_data(train_loader, ptq_batch_size)
print("data loaded")
######################

model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
model.eval()

backend = BackendType.Academic
data = [torch.randn([1, 3, 224,224]) for _ in range(10)]


from mqbench.DotDict import DotDict


### give a adaround config ###
extra_config = {
    'extra_qconfig_dict':{
        'w_fakequantize': 'AdaRoundFakeQuantize',
        'w_qscheme':{
            'bit': 4,
            'symmetry':True,
            'per_channel':False,
            'pot_scale':False
        },
        'a_fakequantize':'LearnableFakeQuantize',
        'a_qscheme':{
            'bit': 4,
            'symmetry': True,
            'per_channel':False,
            'pot_scale':False
        }
    }
}


ptq_reconstruction_config = DotDict(
    pattern='layer',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
    scale_lr= 4.0e-5,                   #? learning rate for learning step size of activation
    warm_up= 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
    weight= 0.01,                       #? loss weight for regularization item
    max_count= 10000,                   #? optimization iteration
    b_range= [20,2],                    #? beta decaying range
    keep_gpu= True,                     #? calibration data restore in gpu or cpu
    round_mode= 'learned_hard_sigmoid', #? ways to reconstruct the weight, currently only support learned_hard_sigmoid
    prob= 1.0,                          #? dropping probability of QDROP, 1.0 for Adaround and BRECQ
)

##############################
#pdb.set_trace()
model = prepare_by_platform(model, backend, extra_config)
model = model.to(device)
enable_calibration(model)
for i, batch in enumerate(calib_data):
    model(batch.to(device))


model = ptq_reconstruction(model, ptq_data, ptq_reconstruction_config)
enable_quantization(model)
top1, top5 = validate_model(val_loader, model, device=device)
logger.info('Quantized model  top1: {}, top5: {}.'.format(top1, top5))