'''
PyTorch数学基础
'''
import torch
'''
PyTorch中的主要函数
'''
# 1、torch.seed()----->用于生成不确定的随机数，返回一个64位的数值
print(torch.seed())
# 2、torch.manual_seed(12)--->设定生成随机数的种子，并返回一个torch.Generator对象
print(torch.manual_seed(12))
# 3、torch.initial_seed()--->返回生成随机数的原始种子值
print(torch.initial_seed())
# 4、返回随机生成器状态
print(torch.get_rng_state())
# 5、设定随机生成器状态
rng_state1 = torch.get_rng_state()
print(rng_state1)
print(torch.set_rng_state(rng_state1*2))
rng_state2 = torch.get_rng_state()
print(rng_state2)
