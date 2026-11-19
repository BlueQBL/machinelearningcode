"""
本文件主要包含数据预处理及常用工具：NumPy、Pandas、Scikit-learn等
"""
'''
1.numpy：是一个用于科学计算的Python库，它提供了大量的函数和数据结构，使得处理大型多维数组和矩阵变得更加简单。
        ndarray是NumPy库的核心数据结构，用于表示多维数组。它是Python编程语言中用于科学计算的基本数据结构之一。
        维度：ndarray可以具有任意数量的维度，从一维（向量）到三维（立体）、四维甚至更高。每个维度的大小称为轴长。
        形状：ndarray对象具有一个描述其维度大小的元组，称为形状(Shape)。例如，二维数组的形状可能是(3, 4)，
            表示有3行4列。
        数据类型：ndarray中的所有元素都必须是相同的数据类型。数据可以是整数、浮点数、复数等。
        内存布局：ndarray在内存中是连续存储的，这意味着它允许高效的数据访问和操作。
        索引和切片：ndarray支持使用方括号进行索引和切片，这允许你访问、修改或操作数组的特定部分。
        广播：ndarray支持广播功能，这是一种机制，允许在不同形状的数组之间进行数学运算。
        数学运算：ndarray支持各种数学运算，如加法、减法、乘法、除法等，这些运算可以逐元素应用于数组中的每个元素。
        函数：NumPy提供了许多内置函数，可以对ndarray执行各种操作，如求和、求平均值、求最大值、求最小值等。
        通用函数：NumPy还提供了通用函数(ufunc)，这些函数可以对ndarray中的每个元素执行操作，而不需要循环。
'''
import numpy as np
# 1.创建NumPy数组
'''
我们可以使用np.array()函数由列表、元组等数据结构中创建NumPy数组。np.array()函数是NumPy库中的一个用于创建数组的函数，
它提供了一种方便的方式来存储和操作数据。具体解释如下：
参数object：这是创建数组的对象，可以是单个值、列表、元组等。这意味着你可以将各种类型的数据传递给np.array()，
    它会尝试将这些数据转换为一个数组。
可选参数dtype：这个参数用于指定创建数组的数据类型。如果没有指定，np.array()会根据传递的对象自动推断数据类型。
返回值：该函数返回一个新的NumPy数组，这个数组可以是一维的（向量），也可以是多维的（矩阵或更高维度的张量）。
np.array()函数非常灵活，可以用来创建各种形状和大小的数组。例如，如果你传递一个列表给np.array()，
    它会创建一个与该列表具有相同元素和顺序的NumPy数组。如果你传递一个整数，它会创建一个包含该整数的一维数组。
    此外，np.array()还支持创建多维数组，这对于科学计算和数据分析非常有用。
'''
# 从列表创建NumPy数组
print("从列表创建NumPy数组：")
arr1 = np.array([1, 2, 3])
print(arr1)
# 从元组创建NumPy数组
arr2 = np.array((1, 2, 3))
print(arr2)
# 2.NumPy数组的属性
'''
NumPy数组有许多有用的属性，例如shape（形状）、dtype（数据类型）和size（元素个数）。
'''
print("NumPy数组的属性：")
arr = np.array([[1, 2], [3, 4]])
print(arr.shape)  # 输出：(2, 2)
print(arr.dtype)  # 输出：int64
print(arr.size)  # 输出：4
# 3.基本操作：NumPy提供了许多用于数组操作的函数，如加法、减法、乘法等。
print("NumPy数组的基本操作：")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# 加法
c = np.add(a, b)
print(c)  # 输出：[5 7 9]
# 减法
d = np.subtract(a, b)
print(d)  # 输出：[-3 -3 -3]
# 乘法
e = np.multiply(a, b)
print(e)  # 输出：[ 4 10 18]
# 4.广播：NumPy支持广播，即在不同形状的数组之间进行数学运算时，会自动扩展较小的数组以匹配较大数组的形状
print("NumPy数组的广播：")
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
# 广播加法
c = a + b
print(c)  # 输出：[3 4 5]
# 5.索引和切片：NumPy数组支持切片和索引操作，可以方便地访问和修改数组的元素。
print("NumPy数组的索引和切片：")
arr = np.array([[1, 2], [3, 4]])
# 获取第一行
row1 = arr[0]
print(row1)  # 输出：[1 2]
# 获取第二列
col2 = arr[:, 1]
print(col2)  # 输出：[2 4]
'''
2.Matplotlib:Matplotlib是一个用于绘制图形的Python库，可用于实现数据的可视化展示。
    Matplotlib提供了丰富的功能，使得创建各种类型的图表变得简单。
'''
# 1.绘制简单折线图:Matplotlib提供了plot()函数，用于绘制折线图
# 导入Matplotlib库中的pyplot模块
import matplotlib.pyplot as plt
print("绘制简单折线图：")
# 定义x和y的值
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# 使用折线图展示x和y的关系
plt.plot(x, y)
# 显示图表
plt.show()
# 2 设置图表标题和坐标轴标签:可以使用title()、xlabel()和ylabel()函数分别设置图表标题、x轴标签和y轴标签
print("设置图表标题和坐标轴标签：")
# 设置matplotlib的字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置matplotlib的坐标轴负号显示为正常字符，避免乱码问题
plt.rcParams['axes.unicode_minus'] = False
# 创建一个简单的折线图
plt.plot(x, y)
# 设置图表标题
plt.title("简单折线图")
# 设置x轴标签
plt.xlabel("x轴")
# 设置y轴标签
plt.ylabel("y轴")
# 显示图表
plt.show()
# 3.绘制散点图:Matplotlib提供了scatter()函数，用于绘制散点图。我们可以传入两个列表作为参数，
#   分别表示x轴和y轴的数据
print("绘制散点图：")
# 定义x和y的值
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# 使用散点图展示x和y的关系
plt.scatter(x, y)
# 显示图表
plt.show()
# 4.绘制柱状图:Matplotlib提供了bar()函数，用于绘制柱状图。我们可以传入两个列表作为参数，
#   分别表示x轴的数据和对应的高度。
print("绘制柱状图：")
# x轴的数据
x = [1, 2, 3, 4, 5]
# y轴的数据（高度）
heights = [2, 4, 6, 8, 10]
# 使用bar()函数绘制柱状图，x和heights分别为x轴和y轴的数据
plt.bar(x, heights)
# 显示图形
plt.show()
# 5 设置图例:我们可以使用legend()函数为图表添加图例。首先，为每个数据系列设置标签，然后调用legend()函数。
print("设置图例：")
# 定义x的值
x = [1, 2, 3, 4, 5]
# 定义y1和y2的值
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]
# 使用折线图展示y1和y2的关系，并添加标签
plt.plot(x, y1, label="数据系列1")
plt.plot(x, y2, label="数据系列2")
# 显示图例
plt.legend()
# 显示图表
plt.show()