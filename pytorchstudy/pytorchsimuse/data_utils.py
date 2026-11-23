"""
本文件主要包含数据预处理及常用工具：NumPy、Pandas、Scikit-learn等
"""
'''
1.numpy：是一个用于科学计算的Python库，它提供了大量的函数和数据结构，使得处理大型多维数组和矩阵变得更加简单。
'''
import numpy as np
# 1.创建NumPy数组
'''
我们可以使用np.array()函数由列表、元组等数据结构中创建NumPy数组。np.array()函数是NumPy库中的一个用于创建数组的函数，
它提供了一种方便的方式来存储和操作数据。具体解释如下：
参数object：这是创建数组的对象，可以是单个值、列表、元组等。
可选参数dtype：这个参数用于指定创建数组的数据类型。如果没有指定，np.array()会根据传递的对象自动推断数据类型。
返回值：该函数返回一个新的NumPy数组，这个数组可以是一维的（向量），也可以是多维的（矩阵或更高维度的张量）。
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
'''
3.数据清洗：数据清洗是数据预处理的一部分，主要是对数据进行审查和校验，以确保其准确性、完整性、一致性和可靠性。
'''
# 1.Pandas的数据结构Series和DataFrame
import pandas as pd
# 创建一个包含整数的Series
print("创建一个包含整数的Series：")
s = pd.Series([1, 2, 3, 4, 5])
# 打印Series的内容
print(s)
print("创建一个包含整数的DataFrame：")
# 创建一个包含姓名、年龄和城市的数据字典
data3 = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
# 将数据字典转换为DataFrame
df = pd.DataFrame(data3)
# 打印DataFrame的内容
print(df)
# Pandas数据清洗示例
print("Pandas数据清洗示例：")
# 1)读取数据
# 读取CSV文件
data = pd.read_csv('表格_20261020.csv')
print(data)
# 读取Excel文件
data1 = pd.read_excel('产品推荐.xlsx')
print(data1)
# 2)查看数据
print(data.head())  # 查看前5行数据
print(data1.head())  # 查看前5行数据
# 3)处理缺失值
# 删除含有缺失值的行
# data.dropna(inplace=True)
# 使用平均值填充缺失值
# data.fillna(data.mean(), inplace=True)
# 4)处理重复值
# 检测重复行
duplicates = data.duplicated()
# 删除重复行
data.drop_duplicates(inplace=True)
# 5)去除异常值
# data = data[(np.abs(stats.zscore(data['column_to_clean'])) < 3)]
# 6)纠正错误
# data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
# 7)数据类型转换
# 将字符串类型的日期转换为日期类型
# data['date'] = pd.to_datetime(data['date'])
# 8)数据筛选和排序
# 筛选年龄大于或等于18岁的记录
filtered_data = df[df['Age'] >= 18]
# 按年龄升序排列
sorted_data = filtered_data.sort_values(by='Age', ascending=True)
print(sorted_data)
# 9)数据合并和连接
'''
merge：根据一个或多个键将两个DataFrame连接在一起。
join：根据一个键将两个DataFrame连接在一起。
concat：在指定轴上连接多个DataFrame。
'''
# 假设有两个数据集：data1和data2
# merged_data = pd.merge(data, data1, on='customer_id')