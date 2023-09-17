import ml_collections

def ACDC_224():
    config = ml_collections.ConfigDict()

    config.pretrain = False
    config.deep_supervision = False


    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [160, 160, 160]         #输入图片的size d,h,w
    # config.hyper_parameter.batch_size = 16
    # config.hyper_parameter.base_learning_rate = 1e-4
    # config.hyper_parameter.model_size = 'Base'
    # config.hyper_parameter.blocks_num = [3, 3, 12, 3]
    config.hyper_parameter.blocks_num = [3, 3, 12, 3]
    # config.hyper_parameter.val_eval_criterion_alpha = 0.9
    # config.hyper_parameter.epochs_num = 500
    config.hyper_parameter.convolution_stem_down = 4
    config.hyper_parameter.channelAttention_reduce = 16
    return config