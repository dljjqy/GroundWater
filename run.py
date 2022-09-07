from utils import *

batch_sizes = [8, 16, 24, 32]
# Single cases

## UNet16 cases
unetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16) for bs in batch_sizes]
unetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16) for bs in batch_sizes]
for case in unetM_2nd_65_16_bs_cases:
    main(case)
for case in unetF_2nd_65_16_bs_cases:
    main(case)
del unetF_2nd_65_16_bs_cases, unetM_2nd_65_16_bs_cases

## UNet16 bc cases
bcunetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16, bc = True) for bs in batch_sizes]
bcunetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16, bc = True) for bs in batch_sizes]
for case in bcunetM_2nd_65_16_bs_cases:
    main(case)
for case in bcunetF_2nd_65_16_bs_cases:
    main(case)
del bcunetF_2nd_65_16_bs_cases, bcunetM_2nd_65_16_bs_cases

## UNet32 cases
unetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 32) for bs in batch_sizes]
unetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 32) for bs in batch_sizes]
for case in unetF_2nd_65_32_bs_cases:
    main(case)
for case in unetM_2nd_65_32_bs_cases:
    main(case)
del unetF_2nd_65_32_bs_cases, unetM_2nd_65_32_bs_cases

## UNet32 bc cases
bcunetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 32, bc = True) for bs in batch_sizes]
bcunetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 32, bc = True) for bs in batch_sizes]
for case in bcunetF_2nd_65_32_bs_cases:
    main(case)
for case in bcunetM_2nd_65_32_bs_cases:
    main(case)
del bcunetF_2nd_65_32_bs_cases, bcunetM_2nd_65_32_bs_cases

# Four cases
## four UNet16 cases
four_unetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16, four = True) for bs in batch_sizes]
four_unetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16, four = True) for bs in batch_sizes]
for case in four_unetM_2nd_65_16_bs_cases:
    main(case)
for case in four_unetF_2nd_65_16_bs_cases:
    main(case)
del four_unetF_2nd_65_16_bs_cases, four_unetM_2nd_65_16_bs_cases

## four UNet16 bc cases
four_bcunetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 16, bc = True, four = True) for bs in batch_sizes]
four_bcunetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 16, bc = True, four = True) for bs in batch_sizes]
for case in four_bcunetM_2nd_65_16_bs_cases:
    main(case)
for case in four_bcunetF_2nd_65_16_bs_cases:
    main(case)
del four_bcunetF_2nd_65_16_bs_cases, four_bcunetM_2nd_65_16_bs_cases

## four UNet32 cases
four_unetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 32, four = True) for bs in batch_sizes]
four_unetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 32, four = True) for bs in batch_sizes]
for case in four_unetM_2nd_65_32_bs_cases:
    main(case)
for case in four_unetF_2nd_65_32_bs_cases:
    main(case)
del four_unetF_2nd_65_32_bs_cases, four_unetM_2nd_65_32_bs_cases

## four UNet16 bc cases
four_bcunetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'UNet', 32, bc = True, four = True) for bs in batch_sizes]
four_bcunetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'UNet', 32, bc = True, four = True) for bs in batch_sizes]
for case in four_bcunetM_2nd_65_32_bs_cases:
    main(case)
for case in four_bcunetF_2nd_65_32_bs_cases:
    main(case)
del four_bcunetF_2nd_65_32_bs_cases, four_bcunetM_2nd_65_32_bs_cases




# attunetF_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'AttUNet', 16) for bs in batch_sizes]
# attunetM_2nd_65_16_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'AttUNet', 16) for bs in batch_sizes]

# attunetF_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'F', 'AttUNet', 32) for bs in batch_sizes]
# attunetM_2nd_65_32_bs_cases = [gen_hyper_dict(65, bs, 2, 'M', 'AttUNet', 32) for bs in batch_sizes]
