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
'''
4.PyTorch矩阵运算
矩阵是一个按照长方阵列排列的复数或实数集合，最早来自方程组的系数及常数所构成的方阵。
矩阵是高等数学中的常见工具，也常见于统计分析等应用数学中。
张量的基本运算方式有逐元素之间的运算
'''
# 1 矩阵的加法
'''
在PyTorch中，矩阵的加法运算有4种实现方法
'''
print("===================矩阵的加法===============")
# 首先创建两个张量
# 创建一个3行4列的随机张量a，元素值在0和1之间
a = torch.rand(3, 4)
# 创建一个长度为4的随机张量b，元素值在0和1之间
b = torch.rand(4)
# 打印张量a的值
print(a)
# 打印张量b的值
print(b)
# 方法1：使用+运算符实现加法
print(a + b)
# 方法2：使用函数torch.add()实现加法。
print(torch.add(a, b))
# 方法3：将结果存储到一个张量
# 创建一个3行4列的空张量c，用于存储结果
c = torch.Tensor(3, 4)
# 将张量a和b逐元素相加，并将结果存储在张量c中
result = torch.add(a, b, out=c)
# 打印张量c的值
print(c)
# 方法4：把一个张量加到另一个张量上
# 将张量a和b逐元素相加，并将结果存储在张量b中
b = b.add(a)
print(b)
# 2 矩阵的减法
'''
在PyTorch中，矩阵的减法与矩阵的加法类似
'''
print("===================矩阵的减法===============")
# 将张量a和b逐元素相减，并将结果打印出来
print(a - b)
# 使用torch.sub函数将张量a和b逐元素相减，并将结果打印出来
print(torch.sub(a, b))
# 创建一个3行4列的空张量c，用于存储结果
c = torch.Tensor(3, 4)
# 使用torch.sub函数将张量a和b逐元素相减，并将结果存储在张量c中
result = torch.sub(a, b, out=c)
# 打印张量c的值
print(c)
# 将张量a和b逐元素相减，并将结果存储在张量b中
b = a.sub(b)
# 打印张量b的值
print(b)
# 3 矩阵的乘法
'''
在PyTorch中，矩阵的乘法与矩阵的加法类似
'''
print("===================矩阵的乘法===============")
# 将张量a和b逐元素相乘，并将结果打印出来
print(a * b)
# 使用torch.mul函数将张量a和b逐元素相乘，并将结果打印出来
print(torch.mul(a, b))
# 创建一个3行4列的空张量c，用于存储结果
c = torch.Tensor(3, 4)
# 使用torch.mul函数将张量a和b逐元素相乘，并将结果存储在张量c中
result = torch.mul(a, b, out=c)
# 打印张量c的值
print(c)
# 将张量a和b逐元素相乘，并将结果存储在张量b中
b = a.mul(b)
# 打印张量b的值
print(b)
# 4 矩阵的除法
'''
在PyTorch中，矩阵的除法与矩阵的加法类似
'''
print("===================矩阵的除法===============")
# 将张量a和b逐元素相除，并将结果打印出来
print(a / b)
# 使用torch.div函数将张量a和b逐元素相除，并将结果打印出来
print(torch.div(a, b))
# 创建一个3行4列的空张量c，用于存储结果
c = torch.Tensor(3, 4)
# 使用torch.div函数将张量a和b逐元素相除，并将结果存储在张量c中
result = torch.div(a, b, out=c)
# 打印张量c的值
print(c)
# 将张量a和b逐元素相除，并将结果存储在张量a中
a = a.div(b)
# 打印张量a的值
print(a)
# 5 矩阵的幂运算
'''
在PyTorch中，还可以实现矩阵的幂运算
'''
print("===================矩阵的幂运算===============")
# 创建一个3行4列的张量f，元素值在0和1之间
f = torch.rand(3, 4)
# 将张量f中的每个元素进行平方运算，并将结果打印出来
print(f.pow(2))
# 6 矩阵的平方根
'''
在PyTorch中，可以实现矩阵的平方根
'''
print("===================矩阵的平方根===============")
# 创建一个3行4列的张量h，元素值在0和1之间
h = torch.rand(3, 4)
# 将张量h中的每个元素进行平方根运算，并将结果打印出来
print(h.sqrt())
# 7 矩阵的对数
'''
在PyTorch中，可以实现矩阵的对数运算
'''
print("===================矩阵的对数===============")
# 创建一个3行4列的张量k，元素值在0和1之间
k = torch.rand(3, 4)
# 将张量k中的每个元素进行以2为底的对数运算，并将结果打印出来
print(torch.log2(k))
# 将张量k中的每个元素进行以10为底的对数运算，并将结果打印出来
print(torch.log10(k))
# 8 其他主要运算
'''
在PyTorch中，还有向下取整、向上取整、四舍五入等张量运算
'''
print("===================其他主要运算===============")
# 创建一个张量j，元素值为3.1415926
j = torch.tensor(3.1415926)
# 将张量j中的元素向下取整，并将结果打印出来
print(j.floor())
# 将张量j中的元素向上取整，并将结果打印出来
print(j.ceil())
# 将张量j中的元素四舍五入取整，并将结果打印出来
print(j.round())
# 在PyTorch中，如果要提取整数部分，可以使用trunc()函数，
# 如果要提取小数部分，可以使用frac()函数（默认保留4位，并进行四舍五入）
print(j.trunc())
print(j.frac())
'''
综合示例：拟合余弦函数曲线
示例说明：本实例使用PyTorch拟合余弦函数曲线，展示预测值和真实值的折线图。
'''
# 实现步骤：
# 1. 导入相关库；
# 导入相关库
# 导入torch.nn模块，用于构建神经网络模型
import torch.nn as nn
# 导入torch.utils.data中的DataLoader和TensorDataset，用于加载数据
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# 导入numpy库，用于进行数值计算
import numpy as np
# 导入matplotlib库，用于绘制图形
import matplotlib
# 导入matplotlib.pyplot模块，用于绘制图形
import matplotlib.pyplot as plt
# 设置matplotlib的字体为SimHei，以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置matplotlib的坐标轴负号显示为正常字符，避免乱码问题
matplotlib.rcParams['axes.unicode_minus'] = False
# 2. 准备拟合数据；
# 生成一个在-2π和2π之间等间距的 400 个点的数列 x
x = np.linspace(-2*np.pi, 2*np.pi, 400)
# 计算数列x的余弦值 y
y = np.cos(x)
# 将 x 扩展为一维数组X，即在列方向上增加一个维度
X = np.expand_dims(x, axis=1)
# 将y整形为 400 行 1 列的矩阵Y
Y = y.reshape(400, -1)
# 将 x 和 y 转换为 torch.Tensor 类型，并组成数据集 dataset
dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y,
dtype=torch.float))
# 创建数据加载器 dataloader，批次大小为 10，数据打乱
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# 3. 设置神经网络，这里使用一个简单的线性结构；
#定义一个名为Net的神经网络模块
class Net(nn.Module):
    def __init__(self):   #构造函数
        super (Net, self).__init__()   #调用父类的构造函数
        self.net = nn.Sequential(      #创建一个序列型神经网络
            # 输入特征数为 1、输出特征数为10的线性层
            nn.Linear (in_features=1, out_features=10),
            nn.ReLU (),                #ReLU激活函数
            nn.Linear(10, 100),        #输入特征数为10、输出特征数为100的线性层
            nn.ReLU(),                 #ReLU激活函数
            nn.Linear(100, 10),        #输入特征数为10、输出特征数为100的线性层
            nn.ReLU(),                 #ReLU激活函数
            #ReLU激活函数
            nn.Linear (10, 1) #输入特征数为10、输出特征数为1的线性层
        )
    def forward (self, input: torch.FloatTensor): #前向传播函数
        return self.net (input) #通过神经网络传递输入数据
#创建Net类的实例
net = Net()
# 4. 设置损失函数和优化器；
# 使用 Adam 优化器对 Net 网络的参数进行优化，学习率为0.001
optim = torch.optim.Adam(Net.parameters(net), lr=0.001)
# 创建均方误差损失函数
Loss = nn.MSELoss()
# 5. 开始训练模型并进行预测，训练1000次；
#遍历1000个训练轮次
for epoch in range (1000):
    loss = None   #初始化损失值为None
    for batch_x, batch_y in dataloader: #遍历数据加载器中每个批次的数据
        y_predict = net(batch_x) #将批次数据输入网络进行预测
        loss = Loss (y_predict, batch_y)  #计算预测结果与真实标签的损失
        optim.zero_grad () #清空优化器的梯度
        loss.backward ()  #反向传播计算梯度
        optim.step ()   #根据梯度更新模型参数
    if (epoch + 1) % 50 == 0:  #每50个轮次打印一次训练信息
        #打印当前轮次和损失值
        print ("训练步骤:{0},模型损失:{1}".format (epoch + 1,loss.item ()))
#使用网络对输入数据X进行预测，并将预测结果存储在predict变量中
predict = net (torch.tensor (X, dtype=torch.float))
# 6. 可视化预测结果和真实值的折线图；
# 创建一个图形，设置图形大小为(12, 7)，分辨率为 160 dpi
plt.figure(figsize=(12, 7), dpi=160)
# 绘制实际值的曲线，标记为"实际值"，标记形状为"X"
plt.plot(x, y, label="实际值", marker="X")
# 绘制预测值的曲线，标记为"预测值"，标记形状为"o"
plt.plot(x, predict.detach().numpy(), label="预测值", marker='o')
# 设置 x 轴标签为"x"，字体大小为 15
plt.xlabel("x", size=15)
# 设置 y 轴标签为"cos(x)"，字体大小为 15
plt.ylabel("cos(x)", size=15)
# 设置 x 轴刻度的字体大小为 15
plt.xticks(size=15)
# 设置 y 轴刻度的字体大小为 15
plt.yticks(size=15)
# 添加图例，字体大小为 15
plt.legend(fontsize=15)
# 显示图形
plt.show()
# 综合练习
# 练习1：综合PyTorch提供的各种张量计算，通过一个程序综合调用
# 创建两个张量
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
# 执行加法操作
z = torch.add(x, y)
print("加法结果：", z)
# 执行减法操作
z = torch.subtract(x, y)
print("减法结果：", z)
# 执行乘法操作
z = torch.multiply(x, y)
print("乘法结果：", z)
# 练习2：通过程序实现计算函数f(x)=x2+2x+1在x=3处的导数
# 背景知识：PyTorch中的自动微分是通过autograd模块实现的。当我们创建一个张量时，可以选择将其设置为requires_grad=True，
# 这样PyTorch就会自动追踪该张量上的所有操作。当我们调用.backward()方法时，PyTorch会自动计算所有相关张量的梯度，
# 并将它们存储在.grad属性中。
# 定义一个张量x，并设置requires_grad=True以跟踪其梯度
x = torch.tensor([3.0], requires_grad=True)
# 定义函数f(x) = x^2 + 2x + 1
y = x ** 2 + 2 * x + 1
# 使用自动微分计算y关于x的导数
y.backward()
# 输出导数值
print("导数值：", x.grad.item())
# 练习3：通过程序实现生成一个包含10个随机数的张量，并计算这些数的均值、中位数、众数、方差和标准差
# 生成一个包含10个随机数的张量
x = torch.randn(10)
# 计算均值
mean = x.mean()
print("均值：", mean.item())
# 计算中位数
median = x.median()
print("中位数：", median.item())
# 计算众数
mode = torch.mode(x)
print("众数：", mode.values.item(), mode.indices.item())
# 计算方差
var = x.var()
print("方差：", var.item())
# 计算标准差
std = x.std()
print("标准差：", std.item())
# 练习4：通过程序实现创建两个形状不同的张量，并演示如何使用矩阵广播机制进行加法、减法、乘法和除法运算
# 创建两个形状不同的张量
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([5, 6])
# 使用矩阵广播机制进行加法、减法、乘法和除法运算
add_result = x + y
subtract_result = x - y
multiply_result = x * y
divide_result = x / y
# 输出结果
print("加法结果：", add_result)
print("减法结果：", subtract_result)
print("乘法结果：", multiply_result)
print("除法结果：", divide_result)