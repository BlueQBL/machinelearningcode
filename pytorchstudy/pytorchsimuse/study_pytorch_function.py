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
# 2.torch.autograd.grad() ----->该函数实现求取梯度
# 函数参数如下：
'''
torch.autograd.grad(outputs,
                    inputs,
                    grad_outputs=None,
                    retain_graph=None,
                    create_graph=False)
'''
# 具体参数说明：
'''
outputs：用于求导的张量，如上例中的loss。
inputs：需要梯度的张量，如上例中的w。
retain_graph：保存计算图。
grad_outputs：多梯度权重。
create_graph：创建导数计算图，用于高阶求导。
'''
# 示例：计算y=x^2的二阶导数的代码的实现
print("===================示例三：计算y=x^2的二阶导数的代码的实现===============")
# 创建一个张量x，值为3.0，并设置requires_grad=True以计算梯度
x = torch.tensor([3.], requires_grad=True)
# 计算x的平方，得到y
y = torch.pow(x, 2)
# 计算y关于x的梯度，create_graph=True表示同时计算二阶导数
grad1 = torch.autograd.grad(y, x, create_graph=True)
# 输出一阶导数：[6.]
print(grad1)
# 计算一阶导数关于x的梯度
grad2 = torch.autograd.grad(grad1[0], x)
# 输出二阶导数：[2.]
print(grad2)
# 3.注意事项
# (1)梯度不能自动清零，在每次反向传播中会叠加
print("===================梯度不能自动清零，在每次反向传播中会叠加===============")
# 创建一个需要梯度的张量w
w = torch.tensor([1.], requires_grad=True)
# 创建一个需要梯度的张量x
x = torch.tensor([2.], requires_grad=True)
# 循环3次
for i in range(3):
    # 将x和w相加，结果存储在a中
    a = torch.add(x, w)
    # 将w和1相加，结果存储在b中
    b = torch.add(w, 1)
    # 将a和b相乘，结果存储在y中
    y = torch.mul(a, b)
    # 计算梯度
    y.backward()
    # 打印w的梯度
    print(w.grad)
# 这会导致我们得不到正确的结果，所以需要手动清零
# 创建一个需要梯度的张量w
print("===================手动清零梯度===============")
w = torch.tensor([1.], requires_grad=True)

# 创建一个需要梯度的张量x
x = torch.tensor([2.], requires_grad=True)

# 循环3次
for i in range(3):
    # 将x和w相加，结果存储在a中
    a = torch.add(x, w)
    # 将w和1相加，结果存储在b中
    b = torch.add(w, 1)
    # 将a和b相乘，结果存储在y中
    y = torch.mul(a, b)
    # 计算梯度
    y.backward()
    # 打印w的梯度
    print(w.grad)
    # 梯度清零
    w.grad.zero_()
# (2)依赖于叶子节点的节点，requires_grad默认为True，
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
# 打印a、b、y是否需要梯度
print(a.requires_grad, b.requires_grad, y.requires_grad)
# (3)叶子节点不可以执行in-place，因为前向传播记录了叶子节点的地址，反向传播需要用到叶子节点的数据时，
# 要根据地址寻找数据，执行in-place操作改变了地址中的数据，梯度求解也会发生错误
print("===================叶子节点不可以执行in-place===============")
# 创建一个需要梯度的张量w
w = torch.tensor([1.], requires_grad=True)
# 创建一个需要梯度的张量x
x = torch.tensor([2.], requires_grad=True)
# 将x和w相加，结果存储在a中
# a = torch.add(x, w)
# 将w和1相加，结果存储在b中
# b = torch.add(w, 1)
# 将a和b相乘，结果存储在y中
# y = torch.mul(a, b)
# 对w进行原地加法操作，即w = w + 1
# w.add_(1)
# in-place操作即原位操作，在原始内存中改变这个数据
a = torch.tensor([1])
print(id(a), a)
# 开辟新的内存地址
a = a + torch.tensor([1])
print(id(a), a)
# in-place操作，地址不变
a += torch.tensor([1])
print(id(a), a)
'''
2.PyTorch统计函数
'''
# 1.求所有元素的积
'''
使用函数torch.prod()求所有元素的积，语法如下:
    torch.prod(input, dim=None, keepdim=False)
torch.prod()是一个PyTorch函数，用于计算张量(Tensor)中所有元素的乘积。
它接受一个张量作为输入，并返回一个标量值，表示输入张量中所有元素的乘积。
【代码说明】
input：输入张量。
dim：指定沿哪个维度进行乘积运算。默认值为None，表示计算整个张量的乘积。
    如果指定了维度，那么结果将是一个降低该维度的张量。
keepdim：布尔值，表示是否保持原始张量的维度。默认值为False，表示不保持原始维度。
    如果设置为True，则结果张量的维度与输入张量相同，但指定的维度大小为1。
'''
print("===================求所有元素的积===============")
# 创建一个2×2的张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a中所有元素的乘积，并将结果赋值给result
result = torch.prod(a)
print(result)  # 输出：24
# 沿着第0维（行）计算张量a中元素的乘积，并将结果赋值给result_dim
result_dim = torch.prod(a, dim=0)
print(result_dim)  # 输出：tensor([3, 8])
# 沿着第0维（行）计算张量a中元素的乘积，并保持原始维度，将结果赋值给result_keepdim
result_keepdim = torch.prod(a, dim=0, keepdim=True)
print(result_keepdim)  # 输出：tensor([[3, 8]])

# 2.求所有元素的和
'''
使用torch.sum()函数求和，该函数对输入的张量数据的某一维度求和，一共有两种格式。
第一种格式:torch.sum(input, dtype=None);计算输入张量input中所有元素的和，返回一个标量值。
第二种格式:torch.sum(input, dim, keepdim=False, dtype=None)
    计算输入张量input在指定维度dim上的和，并返回一个新的张量。
参数说明：
    input：输入一个张量。
    dim：要求和的维度，可以是一个列表。当dim=0时，即第0个维度会缩减，也就是说将N行压缩成一行，
         故相当于对列进行求和；当dim=1时，对行进行求和。
    keepdim：求和之后这个dim的元素个数为1，所以要被去掉，如果要保留这个维度，则要保证keepdim=True。
             如果keepdim为True，则保留原始张量的维度；如果keepdim为False，则不保留原始张量的维度。
'''
print("===================求所有元素的和===============")
# 首先，创建初始张量
# 创建一个2×2的张量a，元素为随机数
a = torch.rand(2, 2)
print(a)
# 设置参数input和dim
a1 = torch.sum(a)  # 计算张量a所有元素的和
a2 = torch.sum(a, dim=(0, 1))  # 沿着第0维和第1维计算张量a的元素和
a3 = torch.sum(a, dim=0)  # 沿着第0维计算张量a的元素和
a4 = torch.sum(a, dim=1)  # 沿着第1维计算张量a的元素和
print(a1)
print(a2)
print(a3)
print(a4)
# 设置参数keepdim
a5 = torch.sum(a, dim=(0, 1), keepdim=True)  # 沿着第0维和第1维计算张量a的元素和，保持原始维度
a6 = torch.sum(a, dim=(0,), keepdim=True)  # 沿着第0维计算张量a的元素和，保持原始维度
a7 = torch.sum(a, dim=(1,), keepdim=True)  # 沿着第1维计算张量a的元素和，保持原始维度
print(a5)
print(a6)
print(a7)
# 3 平均值
'''
使用torch.mean()函数求平均值，该函数对输入的张量数据的某一维度求平均值，参数与torch.sum()函数类似，
也有以下两种格式：     
torch.mean(input, dtype=None)
torch.mean(input, dim, keepdim=False, dtype=None)
'''
print("===================平均值===============")
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a所有元素的平均值，并将结果赋值给a8
a8 = torch.mean(a.float())
# 沿着第0维和第1维计算张量a的元素平均值，并将结果赋值给a9
a9 = torch.mean(a.float(), dim=(0, 1))
# 沿着第0维计算张量a的元素平均值，并将结果赋值给a10
a10 = torch.mean(a.float(), dim=0)
# 沿着第1维计算张量a的元素平均值，并将结果赋值给a11
a11 = torch.mean(a.float(), dim=1)
# 打印a8、a9、a10和a11的值
print(a8)
print(a9)
print(a10)
print(a11)
# 设置参数keepdim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第0维和第1维计算张量a的元素平均值，并保持原始维度，将结果赋值给a12
a12 = torch.mean(a.float(), dim=(0, 1), keepdim=True)
# 沿着第0维计算张量a的元素平均值，并保持原始维度，将结果赋值给a13
a13 = torch.mean(a.float(), dim=(0,), keepdim=True)
# 沿着第1维计算张量a的元素平均值，并保持原始维度，将结果赋值给a14
a14 = torch.mean(a.float(), dim=(1,), keepdim=True)
# 打印a12、a13和a14的值
print(a12)
print(a13)
print(a14)
# 4 最大值
'''
使用torch.max()函数求最大值，参数与torch.sum()函数类似，但是参数dim须为整数
torch.max(input, dtype=None)
torch.max(input, dim, keepdim=False, dtype=None)
'''
print("===================最大值===============")
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a所有元素的最大值，并将结果赋值给a15
a15 = torch.max(a)
# 沿着第0维计算张量a的元素最大值，并将结果赋值给a16
a16 = torch.max(a, dim=0)
# 沿着第1维计算张量a的元素最大值，并将结果赋值给a17
a17 = torch.max(a, dim=1)
# 打印a15、a16和a17的值
print(a15)
print(a16)
print(a17)
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第0维计算张量a的元素最大值，并保持原始维度，将结果赋值给a18
a18 = torch.max(a, 0, keepdim=True)
# 沿着第1维计算张量a的元素最大值，并保持原始维度，将结果赋值给a19
a19 = torch.max(a, 1, keepdim=True)
# 打印a18和a19的值
print(a18)
print(a19)
# 5 最小值
'''
使用torch.min()函数求最小值，参数与torch.max()函数类似
torch.min(input, dtype=None)
torch.min(input, dim, keepdim=False, dtype=None)
'''
print("===================最小值===============")
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a所有元素中的最小值，并将结果赋值给a20
a20 = torch.min(a)
# 沿着第0维计算张量a的元素最小值，并将结果赋值给a21
a21 = torch.min(a, dim=0)
# 沿着第1维计算张量a的元素最小值，并将结果赋值给a22
a22 = torch.min(a, dim=1)
# 打印a20、a21和a22的值
print(a20)
print(a21)
print(a22)
# 设置参数keepdim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第0维计算张量a的元素最小值，并保持原始维度，将结果赋值给a23
a23 = torch.min(a, 0, keepdim=True)
# 沿着第1维计算张量a的元素最小值，并保持原始维度，将结果赋值给a24
a24 = torch.min(a, 1, keepdim=True)
# 打印a23和a24的值
print(a23)
print(a24)
# 6 中位数
'''
使用函数torch.median()求中位数，该函数返回中位数，参数与torch.max()函数类似
torch.median(input, dtype=None)
torch.median(input, dim, keepdim=False, dtype=None)
'''
print("===================中位数===============")
# 设置参数input和dim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a所有元素的中位数，并将结果打印出来
print(torch.median(a))
# 沿着第1维计算张量a的元素中位数，并将结果打印出来
print(torch.median(a, 1))
# 设置参数keepdim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第1维计算张量a的元素中位数，并保持原始维度，将结果赋值给变量median_result
median_result = torch.median(a, 1, keepdim=True)
# 打印median_result的值
print(median_result)
# 7 众数
'''
使用torch.mode()函数求众数，该函数返回众数，参数与torch.max()函数类似:
torch.mode(input, dtype=None)
torch.mode(input, dim, keepdim=False, dtype=None)
'''
print("===================众数===============")
# 设置参数input和dim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 计算张量a所有元素的众数，并将结果打印出来
print(torch.mode(a))
# 沿着第0维计算张量a的元素众数，并将结果打印出来
print(torch.mode(a, 0))
# 设置参数keepdim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第1维计算张量a的元素众数，并保持原始维度，将结果赋值给变量mode_result
mode_result = torch.mode(a, 1, keepdim=True)
# 打印mode_result的值
print(mode_result)
# 8 方差
'''
使用torch.var()函数求方差，该函数返回输入张量中所有元素的方差
torch.var(input, unbiased=True)
torch.var(input, dim, unbiased=True, keepdim=False, *, out=None)
参数说明：
input：输入张量。
dim：要求和的维度，可以是一个列表，
    当dim=0时，即第0个维度会缩减，也就是说将N行压缩成一行，故相当于对列进行求和，
    当dim=1时，对行进行求和。
unbiased：是否使用无偏估计，布尔型。如果unbiased为False，则将通过有偏估计量计算方差，否则将使用“贝塞尔校正”更正。
keepdim：求和之后，这个dim的元素个数为1，所以要被去掉，如果要保留这个维度，则应该将keepdim设置为True。
'''
print("===================方差===============")
# 设置参数input和dim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第1维计算张量a的元素方差，并将结果赋值给变量var_result
var_result = torch.var(a.float(), 1)
# 打印var_result的值
print(var_result)
# 设置参数unbiased
print(torch.var(a.float(), 1, unbiased=False))
# 设置参数keepdim
print(torch.var(a.float(), 1, unbiased=False, keepdim=True))
# 9 标准差
'''
使用torch.std()函数求标准差，该函数返回输入张量中所有元素的标准差，参数与torch.var()函数类似：
torch.std(input, unbiased=True)
torch.std(input, dim, unbiased=True, keepdim=False, *, out=None)
'''
print("===================标准差===============")
# 设置参数input和dim
# 创建一个张量a
a = torch.tensor([[1, 2], [3, 4]])
# 沿着第1维计算张量a的元素标准差，并将结果赋值给变量std_result
std_result = torch.std(a.float(), 1)
# 打印std_result的值
print(std_result)
# 设置参数unbiased
print(torch.std(a.float(), 1, unbiased=False))
# 设置参数keepdim
print(torch.std(a.float(), 1, unbiased=False, keepdim=True))