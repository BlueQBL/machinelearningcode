'''
PyTorch数学基础
'''
import torch
'''
1.PyTorch中的主要函数
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

'''
2.PyTorch自动微分
'''
print("==================================")
# 1、torch.autograd.backward()----->计算梯度
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 5
'''
tensors：待反向传播的目标张量，可以是loss，也可以是张量列表。
grad_tensors：上游梯度，与tensors一一对应；tensors为非标量或多输出张量时必须传入，用于向量‑雅可比乘积。
              普通多loss加权优先直接对loss做加权求和，不建议用该参数。
retain_graph：是否保留反向传播中间缓存。默认反向传播后释放缓存；若要在同一份前向结果上多次执行backward，设置为True。
create_graph：是否为导数构建计算图；设置为True支持高阶导数求解，会增加显存占用。
'''
torch.autograd.backward(
    tensors=y,
    grad_tensors=torch.ones_like(y),  # 上游梯度，和y同shape
    retain_graph=False,
    create_graph=False
)
print(x.grad)
print("===================示例一：线性的一阶导数===============")
# 创建一个需要梯度的张量w
w = torch.tensor([1.], requires_grad=True)
# 创建一个需要梯度的张量x
x = torch.tensor([2.], requires_grad=True)
# 将x和w相加，结果存储在a中
a = torch.add(x, w)
# 将w和1相加，结果存储在b中
b = torch.add(w, 1)
# 将a和b相乘，结果存储在y中
y = torch.mul(a, b)
# 对y进行反向传播计算梯度
y.backward()
# 打印w的梯度
print(w.grad)

print("===================示例二：介绍grad_tensors参数的用法===============")
# 创建一个需要梯度的张量w
w = torch.tensor([1.], requires_grad=True)
# 创建一个需要梯度的张量x
x = torch.tensor([2.], requires_grad=True)
# 将x和w相加，结果存储在a中
a = torch.add(x, w)
# 将w和1相加，结果存储在b中
b = torch.add(w, 1)
# 将a和b相乘，结果存储在y0中
y0 = torch.mul(a, b)
# 将a和b相加，结果存储在y1中
y1 = torch.add(a, b)
# 将y0和y1拼接起来，形成一个新的张量loss
loss = torch.cat([y0, y1], dim=0)
# 定义一个梯度张量grad_t
grad_t = torch.tensor([1., 2.])
# 对loss进行反向传播计算梯度
loss.backward(gradient=grad_t)
# 打印w的梯度
print(w.grad)