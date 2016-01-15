from google.protobuf import text_format
import caffe_pb2 as pb

net = pb.NetParameter()
net_str = str(open('./models/google_net/amazon_to_webcam/train_val.prototxt', 'rb').read())

text_format.Parse(net_str, net)

# lr_mult and decay_mult for inception layers
inceptions = {
    'inception_3a': [[0, 0], [0, 0]],
    'inception_3b': [[0, 0], [0, 0]],
    'inception_4a': [[0, 0], [0, 0]],
    'inception_4b': [[0, 0], [0, 0]],
    'inception_4c': [[0, 0], [0, 0]],
    'inception_4d': [[0.04, 0.08], [1, 0]],
    'inception_4e': [[0.1, 0.2], [1, 0]],
    'inception_5a': [[0.1, 0.2], [1, 0]],
    'inception_5b': [[0.1, 0.2], [1, 0]],
}
# lr_mult and decay_mult for 3 loss layers
losses = {
    'loss1/conv': [[0, 0], [0, 0]],
    'loss1/fc': [[0, 0], [0, 0]],
    'loss1/classifier_office': [[0, 0], [0, 0]],
    'loss2/conv': [[0, 0], [0, 0]],
    'loss2/fc': [[0, 0], [0, 0]],
    'loss2/classifier_office': [[0, 0], [0, 0]],
    'loss3/classifier_office': [[1, 2], [1, 0]],
}
# entropy for entropy layer
entropy = {
    'entropy_loss': [4, 1500],
}
# mmd_lambda for 4 mmd layers
mmds = {
    'mmd3/4d': 2,
    'mmd3/4e': 3,
    'mmd3/5a': 1,
    'mmd3/loss3': 0.3,
}
iter_of_epoch = {
    'mmd3/4d': 56,
    'mmd3/4e': 56,
    'mmd3/5a': 56,
    'mmd3/loss3': 56,
}

# change lr_mult for inception layers
for layer in net.layer:
    name = layer.name
    name_prefic = layer.name.split('/')[0]
    
    if name_prefic in inceptions:
        params = inceptions[name_prefic]
        if len(layer.param) == 2:
            print layer.type
            layer.param[0].lr_mult = params[0][0]
            layer.param[1].lr_mult = params[0][1]
            layer.param[0].decay_mult = params[1][0]
            layer.param[1].decay_mult = params[1][1]
            
    if name in losses:
        params = losses[name]
        if len(layer.param) == 2:
            print 'loss+' + layer.type
            layer.param[0].lr_mult = params[0][0]
            layer.param[1].lr_mult = params[0][1]
            layer.param[0].decay_mult = params[1][0]
            layer.param[1].decay_mult = params[1][1]
        else:
            print 'error'

    if name in mmds:
        mmd_lambda = mmds[name]
        layer.mmd_param.mmd_lambda = mmd_lambda

    if name in iter_of_epoch:
        iters = iter_of_epoch[name]
        layer.mmd_param.iter_of_epoch = iters
    
    if name in entropy:
        layer.loss_weight[0] = entropy[name][0]
        layer.entropy_param.iterations_num = entropy[name][1]

# write to file
output = text_format.MessageToString(net)
out_file = open('./models/google_net/amazon_to_webcam/train_val.prototxt', 'w')
out_file.write(output)
out_file.close()
