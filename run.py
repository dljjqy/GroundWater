from utils import *

batch_sizes = [8, 16, 24, 32]

# unetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16) for bs in batch_sizes]
# unetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16) for bs in batch_sizes]

# bcunetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16, bc = True) for bs in batch_sizes]
# bcunetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16, bc = True) for bs in batch_sizes]

# unetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 32) for bs in batch_sizes]
# unetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 32) for bs in batch_sizes]

# attunetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'AttUNet', 16) for bs in batch_sizes]
# attunetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'AttUNet', 16) for bs in batch_sizes]

# attunetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'AttUNet', 32) for bs in batch_sizes]
# attunetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'AttUNet', 32) for bs in batch_sizes]

if __name__ == '__main__':
    case = gen_hyper_dict(65, 8, 2, 'F', 'AttUNet', 16, lbfgs=False,)
            # ckpt='./lightning_logs/F_65_AttUNet_16_bs8_jac_order2_lr1e-03/version_0/checkpoints/epoch=76-step=19249.ckpt')
    main(case)