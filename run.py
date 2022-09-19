from utils import *

batch_sizes = [8, 16, 24, 32]

# conv_case = gen_hyper_dict(33, 8, 'F', 'UNet', 16, label='conv', conv=True, neu=False,)
# main(conv_case)
# conv_case = gen_hyper_dict(65, 8, 'F', 'UNet', 16, label='conv', conv=True)
# conv_case = gen_hyper_dict(33, 8, 'F', 'UNet', 16, label='conv', conv=True, neu=True)
# main(conv_case)

# conv_case = gen_hyper_dict(65, 8, 'F', 'UNet', 16, label='conv', conv=True, neu=True)


for bs in batch_sizes:
    F_Unet_case = gen_hyper_dict(65, bs, 'F', 'UNet', 16, label='conv', conv=True)
    main(F_Unet_case)

    M_Unet_case = gen_hyper_dict(65, bs, 'M', 'UNet', 16, label='conv', conv=True)
    main(M_Unet_case)

    F_Unet_case = gen_hyper_dict(65, bs, 'F', 'UNet', 16, label='conv', conv=True, neu=True)
    main(F_Unet_case)

    M_Unet_case = gen_hyper_dict(65, bs, 'M', 'UNet', 16, label='conv', conv=True, neu=True)
    main(M_Unet_case)


